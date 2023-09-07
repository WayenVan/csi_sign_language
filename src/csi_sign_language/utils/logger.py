import logging
from time import strftime, localtime
def strtime():
    return strftime('%Y%m%d-%H%M%S', localtime())
    
def build_logger(name, save_directory):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a formatter to specify the log message format
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    # Create a handler to specify where the log messages should go
    file_handler = logging.FileHandler(save_directory)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    

    return logger

    