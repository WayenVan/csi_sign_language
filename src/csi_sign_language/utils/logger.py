import logging

def build_logger(name, save_directory):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a formatter to specify the log message format
    formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s')
    # Create a handler to specify where the log messages should go
    file_handler = logging.FileHandler(save_directory)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

    