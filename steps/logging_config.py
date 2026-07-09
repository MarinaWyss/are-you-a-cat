import logging

# Single source of truth for log formatting across the project. Individual
# modules previously each called ``logging.basicConfig(level=logging.DEBUG)``
# with no format string, which produced inconsistent, timestamp-less output.
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: int = DEFAULT_LOG_LEVEL) -> None:
    """Configures root logging for the project.

    Idempotent: calling this more than once (e.g. from multiple imported
    modules) only configures the root logger a single time.

    Args:
        level (int): Logging level, e.g. ``logging.INFO``.
    """
    global _configured
    if _configured:
        return

    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Returns a module-scoped logger, ensuring logging is configured first.

    Args:
        name (str): Logger name, typically ``__name__``.

    Returns:
        (logging.Logger): Configured logger instance.
    """
    configure_logging()
    return logging.getLogger(name)
