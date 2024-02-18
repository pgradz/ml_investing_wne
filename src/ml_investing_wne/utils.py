import logging
import os

import ml_investing_wne.config as config

file_path = os.path.join(config.package_directory, 'logs')
os.makedirs(file_path, exist_ok=True)
file_dir = os.path.join(file_path, 'app.log')

if not os.path.exists(file_dir):
    # Open the file in 'w' mode and immediately close it, creating an empty file
    with open(file_dir, 'w'):
        pass

def get_logger(file_dir=file_dir):
    # logger settings
    logger = logging.getLogger()
    # You can set a different logging level for each logging handler but it seems you will have
    # to set the logger's level to the "lowest".
    logger.setLevel(logging.INFO)
    stream_h = logging.StreamHandler()
    file_h = logging.FileHandler(file_dir)
    stream_h.setLevel(logging.INFO)
    file_h.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    stream_h.setFormatter(formatter)
    file_h.setFormatter(formatter)
    logger.addHandler(stream_h)
    logger.addHandler(file_h)

    return logger

