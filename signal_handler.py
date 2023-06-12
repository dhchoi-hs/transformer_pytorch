"""

THIS MODULE IS DEPRECATED.
IMPLEMENTED IN A PACKAGE OF HS_AITEAM_PKGS.

handler module on getting kill signal -15.
import this module and catch exception.

Example:
    >>> from signal_handler import SigTermException
    >>> try:
    >>>     while True:
    >>>         # do something...
    >>>         pass
    >>> exception SigTermException:
    >>>     print("sigterm catch")
    >>> exception KeyboardInterrupt:
    >>>     print("KeyboardInterrupt")
"""
import signal


class SigTermException(Exception):
    pass


def _on_get_signal(*args):
    raise SigTermException


signal.signal(signal.SIGTERM, _on_get_signal)  # passed kill signal -15
