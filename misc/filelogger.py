import os
import logging


class FileLogger:
    def __init__(self, output_dir: str, global_rank: int, local_rank: int, name: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.logger = FileLogger.get_logger(
            output_dir, global_rank=global_rank, local_rank=local_rank, name=name)

    def exception(self, *args_, **kwargs):
        return self.logger.exception(*args_, **kwargs)

    @staticmethod
    def get_logger(output_dir: str, global_rank: int, local_rank: int, name: str):
        logger_ = logging.getLogger(name)
        logger_.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        vlog = logging.FileHandler(output_dir + f'/info-{global_rank}.log')
        vlog.setLevel(logging.INFO)
        vlog.setFormatter(formatter)
        logger_.addHandler(vlog)

        eventlog = logging.FileHandler(output_dir + f'/warn-{global_rank}.log')
        eventlog.setLevel(logging.WARN)
        eventlog.setFormatter(formatter)
        logger_.addHandler(eventlog)

        time_formatter = logging.Formatter(
            '%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
        debuglog = logging.FileHandler(
            output_dir + f'/debug-{global_rank}.log')
        debuglog.setLevel(logging.DEBUG)
        debuglog.setFormatter(time_formatter)
        logger_.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        # FIXME:
        # console.setLevel(logging.DEBUG if local_rank == 0 else logging.WARN)
        console.setLevel(logging.DEBUG)
        logger_.addHandler(console)
        return logger_

    def debug(self, *args_):
        self.logger.debug(*args_)

    def warn(self, *args_):
        self.logger.warn(*args_)

    def info(self, *args_):
        self.logger.info(*args_)
