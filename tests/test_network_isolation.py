import contextlib
import socket

import pytest

from tests.support.network import block_network


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_swallowed_network_errors_still_fail(method):
    with (
        pytest.raises(AssertionError, match="Unexpected network attempts"),
        block_network(),
        socket.socket() as sock,
        contextlib.suppress(OSError),
    ):
        getattr(sock, method)(("127.0.0.1", 9))


def test_swallowed_dns_errors_still_fail():
    with (
        pytest.raises(AssertionError, match="Unexpected network attempts"),
        block_network(),
        contextlib.suppress(OSError),
    ):
        socket.getaddrinfo("example.invalid", 443)
