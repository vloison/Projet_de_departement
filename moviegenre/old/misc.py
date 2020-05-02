import os

def create_logger(name, log_dir=None, debug=False):
    """Create a logger.
    Create a logger that logs to log_dir.
    Args:
        name: str. Name of the logger.
        log_dir: str. Path to log directory.
        debug: bool. Whether to set debug level logging.
    Returns:
        logging.logger.
    """

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(process)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format=log_format)
    logger = logging.getLogger(name)
    if log_dir:
        log_file = os.path.join(log_dir, '{}.txt'.format(name))
        file_hdl = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=log_format)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    return logger
