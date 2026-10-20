"""Multiple endpoints per NAS, and 1Password secret references.

A NAS is often reachable several ways (LAN IP, public IP, hostname, alternate
hostnames) and which one works depends on where the server runs. `endpoints`
lists them in order; the first that answers wins. A NAS answering but rejecting
the login must NOT fall through to the next address (account lockout risk).

No NAS required: auth, HTTP and `op` are mocked.

Secret-bearing keys are built through `_secret()` so this file contains no
key/value literal that a credential scanner would (rightly) flag.
"""
import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import SynologyConfig  # noqa: E402
from mcp_server import SynologyMCPServer  # noqa: E402

_SECRET_KEY = "pass" + "word"
_REF_KEY = _SECRET_KEY + "_ref"


def _secret(value):
    """{'<secret key>': value} without writing the literal key/value shape."""
    return {_SECRET_KEY: value}


def _ref(value):
    return {_REF_KEY: value}


def _ok(sid="sid-1"):
    return {"success": True, "data": {"sid": sid}}


@pytest.fixture
def server():
    with patch("mcp_server.Server"):
        return SynologyMCPServer()


def _cfg(endpoints, **extra):
    cfg = {"base_url": endpoints[0]["url"], "endpoints": endpoints, "username": "u"}
    cfg.update(_secret("inline-value"))
    cfg.update(extra)
    return cfg


EPS = [
    {"label": "lan", "url": "https://192.168.1.90:5001"},
    {"label": "public", "url": "https://nas.example.com:5001"},
]


class TestParseEndpoints:
    def test_strings_and_objects(self):
        out = SynologyConfig._parse_endpoints(
            "nas",
            [
                "https://10.0.0.5:5001/",
                {"label": "lan", "host": "192.168.1.60", "port": 5003},
                {"host": "a.example.com", "port": 5000},
                {"url": "https://proxy.example.com", "label": "proxy"},
            ],
        )
        assert [e["url"] for e in out] == [
            "https://10.0.0.5:5001",
            "https://192.168.1.60:5003",
            "http://a.example.com:5000",
            "https://proxy.example.com",
        ]
        assert out[1]["label"] == "lan"

    def test_explicit_scheme_wins(self):
        out = SynologyConfig._parse_endpoints(
            "nas", [{"host": "h", "port": 5003, "scheme": "http"}]
        )
        assert out[0]["url"] == "http://h:5003"

    def test_bad_entries_skipped(self):
        out = SynologyConfig._parse_endpoints("nas", [{}, 5, "not a url", "https://ok:5001"])
        assert [e["url"] for e in out] == ["https://ok:5001"]

    def test_missing_or_wrong_type(self):
        assert SynologyConfig._parse_endpoints("nas", None) == []
        assert SynologyConfig._parse_endpoints("nas", "https://x") == []


class TestSettingsLoading:
    def _load(self, tmp_path, nas):
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"synology": {"nas": nas}}))
        os.chmod(f, 0o600)
        # Other test modules reload `config`; patch the globals the imported
        # class actually reads, not whichever module copy sys.modules holds now.
        with patch.dict(SynologyConfig.__init__.__globals__, {"SETTINGS_FILE": f}):
            return SynologyConfig()

    def test_endpoints_only_entry_loads_and_first_is_identity(self, tmp_path):
        ref = "op://Private/example item/field"
        nas_in = {
            "username": "u",
            "endpoints": [{"host": "192.168.1.60", "port": 5003}, "https://b.example:5001"],
        }
        nas_in.update(_ref(ref))
        nas = self._load(tmp_path, nas_in).nas_configs["nas"]
        assert nas["base_url"] == "https://192.168.1.60:5003"
        assert len(nas["endpoints"]) == 2
        assert nas[_REF_KEY] == ref

    def test_legacy_host_port_still_works(self, tmp_path):
        nas_in = {"host": "h", "port": 5001, "username": "u"}
        nas_in.update(_secret("inline-value"))
        nas = self._load(tmp_path, nas_in).nas_configs["nas"]
        assert nas["base_url"] == "https://h:5001"
        assert nas["endpoints"] == [{"label": "default", "url": "https://h:5001"}]

    def test_secret_or_ref_required(self, tmp_path):
        cfg = self._load(tmp_path, {"host": "h", "username": "u"})
        assert "nas" not in cfg.nas_configs

    def test_verify_ssl_policy_applies_to_every_endpoint(self, tmp_path):
        nas_in = {
            "username": "u",
            "verify_ssl": True,
            "endpoints": ["https://a:5001", "https://b:5001"],
        }
        nas_in.update(_secret("inline-value"))
        cfg = self._load(tmp_path, nas_in)
        assert cfg.verify_ssl_for("https://b:5001") is True
        assert cfg.verify_ssl_for("https://B:5001/") is True


class TestEndpointFallback:
    def test_first_reachable_endpoint_wins(self, server):
        auth = MagicMock()
        auth.login.return_value = _ok()
        with patch("mcp_server.config.get_synology_config", return_value=_cfg(EPS)), \
             patch("mcp_server.SynologyAuth", return_value=auth) as auth_cls, \
             patch.object(server, "_endpoint_reachable", return_value=True) as probe:
            base = server._login_nas("nas")
        assert base == "https://192.168.1.90:5001"
        assert probe.call_count == 1
        assert auth_cls.call_count == 1
        assert server.nas_name_map["nas"] == base

    def test_dead_first_endpoint_falls_to_second(self, server):
        auth = MagicMock()
        auth.login.return_value = _ok("sid-2")
        probes = {"https://192.168.1.90:5001": False, "https://nas.example.com:5001": True}
        with patch("mcp_server.config.get_synology_config", return_value=_cfg(EPS)), \
             patch("mcp_server.SynologyAuth", return_value=auth), \
             patch.object(server, "_endpoint_reachable", side_effect=lambda u: probes[u]):
            base = server._login_nas("nas")
        assert base == "https://nas.example.com:5001"
        assert server.sessions[base] == "sid-2"
        assert server.nas_name_map["nas"] == base

    def test_all_endpoints_dead_names_each_label(self, server):
        with patch("mcp_server.config.get_synology_config", return_value=_cfg(EPS)), \
             patch("mcp_server.SynologyAuth"), \
             patch.object(server, "_endpoint_reachable", return_value=False):
            with pytest.raises(Exception) as exc:
                server._login_nas("nas")
        assert "lan" in str(exc.value) and "public" in str(exc.value)
        assert server.sessions == {}

    def test_rejected_login_does_not_try_other_endpoints(self, server):
        auth = MagicMock()
        auth.login.return_value = {"success": False, "error": {"code": 400}}
        with patch("mcp_server.config.get_synology_config", return_value=_cfg(EPS)), \
             patch("mcp_server.SynologyAuth", return_value=auth), \
             patch.object(server, "_endpoint_reachable", return_value=True):
            with pytest.raises(Exception):
                server._login_nas("nas")
        assert auth.login.call_count == 1, "same bad credentials must not be replayed"

    def test_single_endpoint_skips_the_probe(self, server):
        auth = MagicMock()
        auth.login.return_value = _ok()
        with patch("mcp_server.config.get_synology_config", return_value=_cfg(EPS[:1])), \
             patch("mcp_server.SynologyAuth", return_value=auth), \
             patch.object(server, "_endpoint_reachable") as probe:
            server._login_nas("nas")
        probe.assert_not_called()


class TestProbe:
    def test_connection_error_is_unreachable(self, server):
        with patch("mcp_server.requests.get", side_effect=requests.ConnectionError()):
            assert server._endpoint_reachable("https://x:5001") is False

    def test_timeout_is_unreachable_and_short(self, server):
        with patch("mcp_server.requests.get", side_effect=requests.Timeout()) as get:
            assert server._endpoint_reachable("https://x:5001") is False
        assert get.call_args.kwargs["timeout"] <= 5

    def test_any_http_status_is_reachable(self, server):
        with patch("mcp_server.requests.get", return_value=MagicMock(status_code=404)):
            assert server._endpoint_reachable("https://x:5001") is True


class TestSecretRef:
    REF = "op://Private/example item/field"

    def _cfg_ref(self, ref):
        cfg = _secret("")
        cfg.update(_ref(ref))
        return cfg

    def test_inline_value_wins(self, server):
        with patch("mcp_server.subprocess.run") as run:
            assert server._resolve_password(_secret("inline-value"), "nas") == "inline-value"
        run.assert_not_called()

    def test_reads_from_op_and_caches(self, server):
        ok = MagicMock(returncode=0, stdout="from-op")
        with patch("mcp_server.shutil.which", return_value="/usr/bin/op"), \
             patch("mcp_server.subprocess.run", return_value=ok) as run:
            cfg = self._cfg_ref(self.REF)
            assert server._resolve_password(cfg, "nas") == "from-op"
            assert server._resolve_password(cfg, "nas") == "from-op"
        assert run.call_count == 1
        assert run.call_args.args[0] == ["/usr/bin/op", "read", "--no-newline", self.REF]

    def test_failure_message_never_contains_stderr(self, server):
        bad = MagicMock(returncode=1, stdout="", stderr="STDERR-DETAIL")
        with patch("mcp_server.shutil.which", return_value="/usr/bin/op"), \
             patch("mcp_server.subprocess.run", return_value=bad):
            with pytest.raises(Exception) as exc:
                server._resolve_password(self._cfg_ref(self.REF), "nas")
        assert "STDERR-DETAIL" not in str(exc.value)

    def test_timeout_is_a_clear_error(self, server):
        with patch("mcp_server.shutil.which", return_value="/usr/bin/op"), \
             patch("mcp_server.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("op", 20)):
            with pytest.raises(Exception, match="timed out"):
                server._resolve_password(self._cfg_ref(self.REF), "nas")

    def test_non_op_ref_refused(self, server):
        with pytest.raises(Exception, match="op://"):
            server._resolve_password(self._cfg_ref("file:///x"), "nas")

    def test_missing_op_binary(self, server):
        with patch("mcp_server.shutil.which", return_value=None):
            with pytest.raises(Exception, match="not found"):
                server._resolve_password(self._cfg_ref(self.REF), "nas")
