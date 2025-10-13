class Logger:
    """Simple on/off logger.

    Usage:
        logger = get_logger(enabled=True)
        logger.info("message")
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def info(self, *args, **kwargs):
        if self.enabled:
            print(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.enabled:
            print(*args, **kwargs)

    def error(self, *args, **kwargs):
        # always print errors
        print(*args, **kwargs)


def get_logger(enabled: bool = True) -> Logger:
    return Logger(enabled=enabled)
