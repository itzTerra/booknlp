class Logger:
    """Simple on/off logger that records messages into an in-memory buffer.

    Instead of printing to stdout/stderr, messages are appended to a module-
    level list which can later be attached to result objects for debugging.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def _format_args(self, *args, **kwargs) -> str:
        parts = [str(a) for a in args]
        # ignore kwargs formatting complexity for now
        return " ".join(parts)

    def info(self, *args, **kwargs):
        if self.enabled:
            _record_log("INFO", self._format_args(*args, **kwargs))

    def warning(self, *args, **kwargs):
        if self.enabled:
            _record_log("WARN", self._format_args(*args, **kwargs))

    def error(self, *args, **kwargs):
        # record errors as well (don't print)
        _record_log("ERROR", self._format_args(*args, **kwargs))


# Module-level in-memory log buffer
_LOGS = []


def _record_log(level: str, message: str) -> None:
    _LOGS.append({"level": level, "message": message})


def get_logger(enabled: bool = True) -> Logger:
    return Logger(enabled=enabled)


def get_logs() -> list:
    """Return the collected logs as a list of dicts.

    Each entry has shape: { 'level': str, 'message': str }
    """
    return list(_LOGS)


def clear_logs() -> None:
    """Clear the module-level log buffer."""
    _LOGS.clear()
