import os
import logging

import ml_investing_wne.config as config


def get_logger():
    # logger settings
    logger = logging.getLogger()
    # You can set a different logging level for each logging handler but it seems you will have
    # to set the logger's level to the "lowest".
    logger.setLevel(logging.INFO)
    stream_h = logging.StreamHandler()
    dir = os.path.join(config.package_directory, 'logs')
    file = os.path.join(dir, 'app.log')
    if not os.path.exists(dir):
        os.makedirs(dir)
        open(file, 'w').close()
    file_h = logging.FileHandler(file)
    stream_h.setLevel(logging.INFO)
    file_h.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    stream_h.setFormatter(formatter)
    file_h.setFormatter(formatter)
    logger.addHandler(stream_h)
    logger.addHandler(file_h)

    return logger
