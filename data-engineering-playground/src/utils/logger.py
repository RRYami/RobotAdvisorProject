import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name, log_level=logging.INFO):
    """
    Setup logger for the application

    Args:
        name (str): Name of the logger
        log_level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger object
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parents[2] / "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{current_date}_{name}.log"

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Create console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
