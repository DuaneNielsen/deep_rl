class Logger:
    def __init__(self):
        self.log = {}
        self.writers = []
        self.run_dir = None

    def write(self):
        for writer in self.writers:
            writer(self)
        self.log = {}


logger = Logger()


def init(run_dir):
    logger.run_dir = run_dir


def log(log_dict):
    logger.log.update(log_dict)


def write():
    logger.write()