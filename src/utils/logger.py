import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from icecream import ic
import sys
import inspect
import os


class IcyLogger:
    """
    Custom logger that supports IceCream for contextual debugging and traditional log levels.
    """
    def __init__(self, name=None, enable_logging=True, log_file="./logs/app.log"):
        """
        Initialize the CustomLogger class.
        :param name: Name of the logger, typically the module name (e.g., __name__).
        :param enable_logging: Whether to enable logging by default.
        :param log_file: Path to the log file for saving debug logs.
        """
        self.name = name or self._get_caller_name()
        self.enable_logging = enable_logging
        self.log_file = log_file

        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Configure the traditional logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)  # Capture all log levels
        self.logger.propagate = False  # Prevent duplicate logs in the root logger
        self._setup_handlers()

        # Configure IceCream
        self._configure_ic()

    def _setup_handlers(self):
        """
        Set up log handlers for file and console output.
        """
        self.logger.handlers.clear()

        # File Handler for DEBUG logs
        file_handler = ConcurrentRotatingFileHandler(
            self.log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
        self.logger.addHandler(file_handler)

        # Console Handler for INFO logs only
        class InfoFilter(logging.Filter):
            def filter(self, record):
                return record.levelno == logging.INFO

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(InfoFilter())  # Filter to only include INFO logs
        console_handler.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
        self.logger.addHandler(console_handler)


    def _configure_ic(self):
        """
        Configure the IceCream logger for contextual debugging.
        """
        ic.configureOutput(
            prefix=f"[{self.name}] [DEBUG] ",
            includeContext=True,  # Includes filename and line number
            outputFunction=self._log_to_stdout
        )
        if not self.enable_logging:
            ic.disable()

    def _log_to_stdout(self, message):
        """
        Redirect IceCream output to the logger's debug stream.
        """
        self.logger.debug(message)

    @staticmethod
    def _get_caller_name():
        """
        Get the name of the caller module.
        """
        frame = inspect.currentframe().f_back.f_back
        return os.path.basename(frame.f_code.co_filename).replace(".py", "")

    def set_name(self, name):
        """
        Dynamically update the logger's name.
        """
        self.name = name
        self.logger.name = name
        self._configure_ic()

    def debug(self, *args, **kwargs):
        """
        Log a debug message.
        """
        self.logger.debug(self._format_message(*args, **kwargs))

    def info(self, *args, **kwargs):
        """
        Log an info message.
        """
        self.logger.info(self._format_message(*args, **kwargs))

    def warning(self, *args, **kwargs):
        """
        Log a warning message.
        """
        self.logger.warning(self._format_message(*args, **kwargs))

    def error(self, *args, exception=None, **kwargs):
        """
        Log an error message with optional exception details.
        """
        if exception:
            exception_message = "".join(logging.traceback.format_exception(None, exception, exception.__traceback__))
            kwargs["traceback"] = exception_message
        self.logger.error(self._format_message(*args, **kwargs))

    def log(self, *args, **kwargs):
        """
        Fallback method for generic logging.
        """
        self.debug(*args, **kwargs)

    @staticmethod
    def _format_message(*args, **kwargs):
        """
        Format the log message.
        """
        args_message = " ".join(map(str, args))
        kwargs_message = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{args_message} {kwargs_message}".strip()


# Global logger function for convenience
def get_logger(name=None, log_file="./logs/app.log"):
    """
    Create or retrieve a CustomLogger instance.
    :param name: Logger name (e.g., __name__ of the module).
    :param log_file: Path to the log file for saving debug logs.
    :return: Configured CustomLogger instance.
    """
    return IcyLogger(name=name, log_file=log_file)
