from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from matplotlib.figure import Figure
from pandas import DataFrame

from msd.configurations.msd_component import MSDComponent


class BaseLogger(ABC, MSDComponent):
    """
    Abstract base class for logging backends.

    Subclasses should implement support for logging scalar values, structured data,
    plots, audio, and debug messages, typically to disk, console, or experiment trackers.
    """

    def __init__(self):
        super(BaseLogger, self).__init__()

    @abstractmethod
    def debug(self, msg: str) -> None:
        """Log a debug message."""
        pass

    @abstractmethod
    def info(self, msg: str) -> None:
        """Log an informational message."""
        pass

    @abstractmethod
    def warning(self, msg: str) -> None:
        """Log a warning message."""
        pass

    @abstractmethod
    def error(self, msg: str) -> None:
        """Log an error message."""
        pass

    @abstractmethod
    def log(self, name: str, data: Any, step: Optional[int] = None) -> None:
        """
        Log a single scalar or value.

        :param name: Metric name.
        :param data: Value to log (scalar, string, etc.).
        :param step: Optional global step (e.g., epoch).
        """
        pass

    @abstractmethod
    def log_dict(self, name: str, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log a dictionary of metrics under a given name.

        :param name: Prefix for all keys.
        :param data: Dictionary of key-value pairs.
        :param step: Optional global step.
        """
        pass

    @abstractmethod
    def log_table(self, name: str, data: DataFrame, step: Optional[int] = None) -> None:
        """
        Log a table (e.g., metrics or results).

        :param name: Table identifier.
        :param data: A pandas DataFrame.
        :param step: Optional global step.
        """
        pass

    @abstractmethod
    def log_file(self, name: str, file_path: str) -> None:
        """
        Log a reference to a file (e.g., artifact or result).

        :param name: Artifact name.
        :param file_path: Path to file.
        """
        pass

    @abstractmethod
    def log_audio(self, name: str, data: Any, sample_rate: int, step: Optional[int] = None) -> None:
        """
        Log an audio waveform.

        :param name: Audio tag.
        :param data: Audio data (e.g., numpy array or tensor).
        :param sample_rate: Sample rate in Hz.
        :param step: Optional step.
        """
        pass

    @abstractmethod
    def plot(self, name: str, data: Figure, step: Optional[int] = None) -> None:
        """
        Log a matplotlib figure.

        :param name: Plot identifier.
        :param data: Matplotlib figure object.
        :param step: Optional global step.
        """
        pass


class CompositeLogger(BaseLogger):
    """
    Logger that delegates to multiple loggers.

    Useful for logging simultaneously to console, files, and external tools (e.g. WandB, TensorBoard).
    """

    def __init__(self, loggers: List[BaseLogger]):
        """
        :param loggers: A list of logger instances to broadcast to.
        """
        super(CompositeLogger, self).__init__()
        self.loggers = loggers

    def debug(self, msg: str) -> None:
        for logger in self.loggers:
            logger.debug(msg)

    def info(self, msg: str) -> None:
        for logger in self.loggers:
            logger.info(msg)

    def warning(self, msg: str) -> None:
        for logger in self.loggers:
            logger.warning(msg)

    def error(self, msg: str) -> None:
        for logger in self.loggers:
            logger.error(msg)

    def log(self, name: str, data: Any, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log(name, data, step)

    def log_dict(self, name: str, data: Dict[str, Any], step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_dict(name, data, step)

    def log_table(self, name: str, data: DataFrame, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_table(name, data, step)

    def log_file(self, name: str, file_path: str) -> None:
        for logger in self.loggers:
            logger.log_file(name, file_path)

    def log_audio(self, name: str, data: Any, sample_rate: int, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log_audio(name, data, sample_rate, step)

    def plot(self, name: str, data: Figure, step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.plot(name, data, step)