"""The tests never leave the machine, and leave nothing behind: any socket opened while they run raises, and all
their scratch goes in one folder that is removed when they end."""
import atexit
import os
import shutil
import socket
import tempfile

os.environ["EUNOE_OFFLINE"] = "1"
tempfile.tempdir = tempfile.mkdtemp(prefix="eunoe-tests-")
atexit.register(shutil.rmtree, tempfile.tempdir, ignore_errors=True)


class _NoNetwork(socket.socket):
    def connect(self, *args, **kwargs):
        raise AssertionError("a test tried to open a network connection")


socket.socket = _NoNetwork
