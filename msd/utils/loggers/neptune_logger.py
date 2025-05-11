import os
from os import path as osp
from pathlib import Path
from typing import Any, Dict, Optional

import neptune
from matplotlib.figure import Figure
from neptune.types import File
from pandas import DataFrame
from scipy.io import wavfile

from msd.utils.loggers.base_logger import BaseLogger

class NeptuneLogger(BaseLogger):
    """
    Logger that integrates with Neptune.ai for experiment tracking.

    Supports logging scalars, dicts, plots, tables, audio, and files to a remote Neptune run.
    """

    def __init__(
            self,
            project: str,
            tags: Optional[Dict[str, Any]] = None,
            capture_stdout: bool = False,
            capture_stderr: bool = False,
            api_token: Optional[str] = None
    ):
        """
        :param project: Project name in format "workspace/project-name".
        :param tags: Optional tags to attach to the run.
        :param capture_stdout: If True, capture print output.
        :param capture_stderr: If True, capture error output.
        :param api_token: Optional API token for authentication.
        """
        super(NeptuneLogger, self).__init__()
        if api_token is None:
            api_token = Path.home().joinpath('.neptune', 'token.txt')
            api_token = api_token.read_text().strip()
        if isinstance(tags, str):
            tags = [tags]
        self.run = neptune.init_run(
            project=project,
            api_token=api_token,
            tags=list(tags) if tags is not None else None,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr
        )


    def debug(self, msg: str) -> None:
        """No-op in NeptuneLogger."""
        pass

    def info(self, msg: str) -> None:
        """No-op in NeptuneLogger."""
        pass

    def warning(self, msg: str) -> None:
        """No-op in NeptuneLogger."""
        pass

    def error(self, msg: str) -> None:
        """No-op in NeptuneLogger."""
        pass

    def log(self, name: str, data: Any, step: Optional[int] = None) -> None:
        """
        Log a scalar or basic value to Neptune.

        :param name: Metric name.
        :param data: Value to log.
        :param step: Optional step index.
        """
        self.run[name].log(data, step=step)


    def log_dict(self, name: str, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log a dictionary of values under a common name prefix.

        Nested structures are converted to strings before logging.

        :param name: Prefix for metrics.
        :param data: Dictionary of key-value pairs.
        :param step: Optional step.
        """
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                v = str(v)
            self.log(f'{name}/{k}', v, step)

    def log_table(self, name: str, data: DataFrame, step: Optional[int] = None) -> None:
        """
        Log a table (DataFrame) as HTML.

        :param name: Table name.
        :param data: Pandas DataFrame.
        :param step: Optional step (used in key name).
        """
        self.run[f'{name}/{step}'].upload(File.as_html(data))

    def log_file(self, name: str, file_path: str) -> None:
        """
        Upload a file to Neptune.

        :param name: Artifact name.
        :param file_path: Path to the file.
        """
        self.run[name].upload(osp.join(file_path))


    def log_audio(self, name: str, data: Any, sample_rate: int, step: Optional[int] = None) -> None:
        """
        Log audio data by writing it to a temporary WAV file and uploading it.

        :param name: Audio identifier.
        :param data: Audio waveform (numpy array).
        :param sample_rate: Sample rate in Hz.
        :param step: Optional global step.
        """
        _name = name.replace('/', '_').replace('\\', '_')
        tmp_name = f'{_name}_tmp.wav'
        wavfile.write(tmp_name, sample_rate, data)
        self.run[name].upload(tmp_name)
        self.run.wait()
        os.remove(tmp_name)

    def plot(self, name: str, fig: Figure, step: Optional[int] = None) -> None:
        """
        Upload a matplotlib figure to Neptune.

        :param name: Plot name.
        :param fig: Matplotlib figure.
        :param step: Optional step.
        """
        self.run[name].log(fig)
