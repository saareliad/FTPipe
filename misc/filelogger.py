import os
import logging

# TODO: completely replace this, didn't pay attention to logging so far.


class FileLogger:
    def __init__(self, output_dir: str, global_rank: int, local_rank: int, name: str, world_size: int, name_prefix=''):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.logger = FileLogger.get_logger(
            output_dir, global_rank=global_rank, local_rank=local_rank,
            name=name, world_size=world_size, name_prefix=name_prefix)

    def exception(self, *args_, **kwargs):
        return self.logger.exception(*args_, **kwargs)

    @staticmethod
    def get_logger(output_dir: str, global_rank: int, local_rank: int, name: str,  world_size: int, name_prefix=''):
        logger_ = logging.getLogger(name)
        logger_.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        def get_name(u):
            curr_name = f'{name_prefix}-{u}-{global_rank}.log'
            curr_name = os.path.join(output_dir, curr_name)
            return curr_name

        vlog = logging.FileHandler(get_name('info'))
        vlog.setLevel(logging.INFO)
        vlog.setFormatter(formatter)
        logger_.addHandler(vlog)

        eventlog = logging.FileHandler(get_name('warn'))
        eventlog.setLevel(logging.WARN)
        eventlog.setFormatter(formatter)
        logger_.addHandler(eventlog)

        time_formatter = logging.Formatter(
            '%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
        debuglog = logging.FileHandler(
            get_name('debug'))
        debuglog.setLevel(logging.DEBUG)
        debuglog.setFormatter(time_formatter)
        logger_.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        # FIXME:
        # console.setLevel(logging.DEBUG if local_rank == world_size - 1 else logging.WARN)
        # console.setLevel(logging.DEBUG if local_rank == 0 else logging.WARN)
        console.setLevel(logging.DEBUG)
        logger_.addHandler(console)
        return logger_

    def debug(self, *args_):
        self.logger.debug(*args_)

    def warning(self, *args_):
        self.logger.warning(*args_)

    def info(self, *args_):
        self.logger.info(*args_)
