import socket
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def block_network():
    attempts = []
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def checked_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(str(address))
            raise OSError("Network disabled in tests")
        return connect(sock, address)

    def checked_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(str(address))
            raise OSError("Network disabled in tests")
        return connect_ex(sock, address)

    def checked_lookup(*args, **kwargs):
        attempts.append(str(args[0]))
        raise OSError("DNS disabled in tests")

    with (
        patch.object(socket.socket, "connect", checked_connect),
        patch.object(socket.socket, "connect_ex", checked_connect_ex),
        patch.object(socket, "getaddrinfo", checked_lookup),
    ):
        try:
            yield
        finally:
            assert not attempts, f"Unexpected network attempts: {attempts}"
