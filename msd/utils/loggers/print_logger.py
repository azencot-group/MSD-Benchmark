import logging
from os import path as osp
from typing import Any, Dict, Optional

from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.io import wavfile
from tabulate import tabulate

from msd.utils.loading_utils import init_directories
from msd.utils.loggers.base_logger import BaseLogger

class PrintLogger(BaseLogger):
    """
    Logger that prints logs to the console using Python's logging module.
    Optionally displays plots inline.
    """

    def __init__(self, name: str, show: bool = False, log_level: int = logging.DEBUG):
        """
        :param name: Logger name.
        :param show: If True, display matplotlib plots using `fig.show()`.
        :param log_level: Logging level (default: logging.DEBUG).
        """
        super().__init__()
        self.name = name
        self.show = show
        self.log_level = log_level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setLevel(self.log_level)
        sh.setFormatter(self.formatter)
        self.logger.addHandler(sh)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def log(self, name: str, data: Any, step: Optional[int] = None) -> None:
        self.logger.info(f'{name} ({step}): {data}')

    def log_dict(self, name: str, data: Dict[str, Any], step: Optional[int] = None) -> None:
        self.log(name, data, step)

    def log_table(self, name: str, data: DataFrame, step: Optional[int] = None) -> None:
        """
        Print a DataFrame as a formatted table.

        :param name: Identifier name.
        :param data: DataFrame to log.
        :param step: Optional global step.
        """
        self.logger.info(f'{name} ({step}):\n{tabulate(data, headers="keys", tablefmt="psql")}')

    def log_file(self, name: str, file_path: str) -> None:
        """
        Print the contents of a text file.

        :param name: Label for the file.
        :param file_path: Path to file.
        """
        with open(file_path, 'r') as f:
            self.logger.info(f'{name}: {f.read()}')

    def log_audio(self, name: str, data: Any, sample_rate: int, step: Optional[int] = None) -> None:
        """
        No-op in base PrintLogger. Implemented in FileLogger.
        """
        pass

    def plot(self, name: str, fig: Figure, step: Optional[int] = None) -> None:
        """
        Display a matplotlib figure if `show=True`.

        :param name: Plot name.
        :param fig: Matplotlib figure.
        :param step: Optional global step.
        """
        if self.show:
            fig.show()

class FileLogger(PrintLogger):
    """
    Logger that writes logs, plots, audio, and tables to disk.

    Inherits printing functionality from PrintLogger and extends it
    with persistent file output.
    """

    def __init__(self, name: str, log_path: str, show: bool = False, log_level: int = logging.DEBUG):
        """
        :param name: Logger name.
        :param log_path: Directory to save log outputs.
        :param show: If True, display plots inline.
        :param log_level: Logging level.
        """
        super().__init__(name, show, log_level)
        self.path = log_path
        init_directories(self.path)

        fh = logging.FileHandler(osp.join(self.path, f'{self.name}.log'))
        fh.setLevel(self.log_level)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)


    def plot(self, name: str, fig: Figure, step: Optional[int] = None) -> None:
        """
        Save figure to PNG file in the log directory.

        :param name: Plot name.
        :param fig: Matplotlib figure.
        :param step: Optional global step for naming.
        """
        super().plot(name, fig)
        _name = name.replace('/', '_').replace('\\', '_')
        fig.savefig(osp.join(self.path, f'{step}_{_name}.png'), dpi=300)

    def log_audio(self, name: str, data: Any, sample_rate: int, step: Optional[int] = None) -> None:
        """
        Save audio data to .wav file in the log directory.

        :param name: Audio identifier.
        :param data: Audio array.
        :param sample_rate: Sampling rate in Hz.
        :param step: Optional global step for naming.
        """
        super().log_audio(name, data, sample_rate, step)
        _name = name.replace('/', '_').replace('\\', '_')
        wavfile.write(osp.join(self.path, f'{step}_{_name}.wav'), sample_rate, data)

    def log_table(self, name: str, data: DataFrame, step: Optional[int] = None) -> None:
        """
        Save table (DataFrame) to CSV and also print it.

        :param name: Table name.
        :param data: Pandas DataFrame to log.
        :param step: Optional global step.
        """
        super().log_table(name, data, step)
        _name = name.replace('/', '_').replace('\\', '_')
        data.to_csv(osp.join(self.path, f'{step}_{_name}.csv'))
