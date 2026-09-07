import socket

import pytest

from tests._network import block_network


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_swallowed_network_errors_still_fail(method):
    with pytest.raises(AssertionError, match="Unexpected network attempts"):
        with block_network(), socket.socket() as sock:
            try:
                getattr(sock, method)(("127.0.0.1", 9))
            except OSError:
                pass


def test_swallowed_dns_errors_still_fail():
    with pytest.raises(AssertionError, match="Unexpected network attempts"):
        with block_network():
            try:
                socket.getaddrinfo("example.invalid", 443)
            except OSError:
                pass
