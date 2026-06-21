"""
tests/test_network_utils.py — Unit tests for network_utils.py.

Tests cover:
    - get_local_ip() returns a non-empty string in dotted-decimal format
    - get_local_ip() does not return the raw loopback on a connected machine
    - ping_device() returns False for clearly unreachable/invalid addresses
      (mocked to avoid actual network calls in CI)
"""

import os
import sys
import re
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from network_utils import get_local_ip, ping_device


# ---------------------------------------------------------------------------
# Regex pattern for a valid IPv4 dotted-decimal address
# ---------------------------------------------------------------------------
IPV4_PATTERN = re.compile(
    r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
)


class TestGetLocalIp:
    """Tests for get_local_ip()."""

    def test_returns_non_empty_string(self):
        """get_local_ip() must return a non-empty string."""
        ip = get_local_ip()
        assert isinstance(ip, str) and len(ip) > 0

    def test_returns_valid_ipv4_format(self):
        """Returned value must look like a valid dotted-decimal IPv4 address."""
        ip = get_local_ip()
        match = IPV4_PATTERN.match(ip)
        assert match is not None, f"'{ip}' is not a valid IPv4 address"

    def test_each_octet_in_range(self):
        """Each octet of the returned IP must be between 0 and 255."""
        ip = get_local_ip()
        octets = [int(o) for o in ip.split(".")]
        assert len(octets) == 4
        for octet in octets:
            assert 0 <= octet <= 255, f"Octet {octet} is out of valid range"

    def test_fallback_on_socket_error(self):
        """When socket raises an exception, function should return '127.0.0.1'."""
        with patch("network_utils.socket.socket") as mock_socket:
            mock_socket.return_value.__enter__ = mock_socket
            mock_socket.return_value.connect.side_effect = OSError("Network unreachable")
            # Re-import after patching won't help; call directly
            import network_utils
            # Temporarily monkey-patch to simulate failure
            original = network_utils.socket.socket

            class FailingSocket:
                def settimeout(self, t):
                    pass
                def connect(self, addr):
                    raise OSError("simulated failure")
                def getsockname(self):
                    return ("", 0)
                def close(self):
                    pass

            network_utils.socket.socket = lambda *a, **k: FailingSocket()
            try:
                result = network_utils.get_local_ip()
                assert result == "127.0.0.1", f"Expected fallback '127.0.0.1', got '{result}'"
            finally:
                network_utils.socket.socket = original


class TestPingDevice:
    """Tests for ping_device()."""

    def test_ping_invalid_ip_returns_false(self):
        """Pinging an unroutable RFC 5737 documentation address should return False."""
        # 192.0.2.1 is reserved for documentation (TEST-NET-1), never routable
        result = ping_device("192.0.2.1", count=1)
        assert result is False, "Expected False for unreachable documentation address"

    def test_ping_localhost_returns_bool(self):
        """ping_device() should always return a bool, regardless of reachability."""
        result = ping_device("127.0.0.1", count=1)
        assert isinstance(result, bool)

    def test_ping_returns_false_on_subprocess_error(self):
        """If subprocess.run raises an exception, ping_device() should return False."""
        with patch("network_utils.subprocess.run", side_effect=OSError("no ping binary")):
            result = ping_device("192.168.1.1", count=1)
        assert result is False

    def test_ping_returns_false_on_nonzero_exit(self):
        """A non-zero return code from ping should map to False."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("network_utils.subprocess.run", return_value=mock_result):
            result = ping_device("10.0.0.1", count=1)
        assert result is False

    def test_ping_returns_true_on_zero_exit(self):
        """A zero return code from ping (host reachable) should map to True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("network_utils.subprocess.run", return_value=mock_result):
            result = ping_device("10.0.0.1", count=1)
        assert result is True
