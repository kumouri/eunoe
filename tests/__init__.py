"""The tests never leave the machine: any socket opened while they run raises."""
import os
import socket

os.environ["EUNOE_OFFLINE"] = "1"


class _NoNetwork(socket.socket):
    def connect(self, *args, **kwargs):
        raise AssertionError("a test tried to open a network connection")


socket.socket = _NoNetwork
