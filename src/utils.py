
def set_seeds(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_logger():
    import logging
    from rich.logging import RichHandler
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log
LOG = get_logger()
