import contextlib
import io
import unittest

from eunoe.cli import main


class CliTest(unittest.TestCase):
    def test_help_exits_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
