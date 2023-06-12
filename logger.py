"""

THIS MODULE IS DEPRECATED.
IMPLEMENTED IN A PACKAGE OF HS_AITEAM_PKGS.

"""
import logging


_global_logger = None


def init_logger(filename=None, level=logging.DEBUG):
    global _global_logger
    if _global_logger is not None:
        return
    # TODO: implement file rotate 
    formatter = logging.Formatter(
        '%(asctime)s(%(process)d)(%(funcName)s)[%(levelname)s] %(message)s')
    logger = logging.getLogger()
    streamer = logging.StreamHandler()
    streamer.setFormatter(formatter)
    logger.addHandler(streamer)
    if filename:
        handler = logging.FileHandler(filename, encoding='utf8')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)

    _global_logger = logger


def get_logger():
    global _global_logger
    if _global_logger is None:
        init_logger()
        
    return _global_logger
