import logging
import sys
from pathlib import Path


def setup_logging(path):
    logging.basicConfig(filename=path,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)


def logging_conf(path: Path, logname: str) -> logging.Logger:
    """
    Configure logging to a file.

    :param path: The path to the directory where the log file should be created.
    :type path: pathlib.Path
    :param logname: The name of the log file.
    :type logname: str
    :raises TypeError: If path is not a pathlib.Path object, or logname is not a string.
    :raises OSError: If the log file cannot be created or opened.
    :return: The logger object.
    :rtype: logging.Logger
    """
    # Input validation
    if not isinstance(path, Path):
        raise TypeError('path must be a pathlib.Path object')
    if not isinstance(logname, str):
        raise TypeError('logname must be a string')

    # Set up logging to a file
    log_path = path / logname
    try:
        setup_logging(log_path)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # create logger for "my-setup" and add file handler
        logger = logging.getLogger("my-setup")
        logger.setLevel(level=logging.DEBUG)
        
        # Add filter to exclude matplotlib logs
        matplotlib_logger = logging.getLogger('matplotlib')
        matplotlib_logger.setLevel(logging.WARNING)
        logger.addFilter(lambda record: "matplotlib" not in record.name)
        astropy_logger = logging.getLogger('astropy')
        astropy_logger.setLevel(logging.WARNING)
        logger.addFilter(lambda record: "astropy" not in record.name)
        
        logger.addHandler(fh)
        logger.propagate = False  # stop the logs from propagating to the root logger

        # create console handler with a higher INFO log level and add to logger
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    except OSError as e:
        raise OSError(f'Unable to create or open log file {log_path}: {e}') from None

    logger.debug("Start session")
    return logger
