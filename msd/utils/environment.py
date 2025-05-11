import os
import platform
import subprocess

import numpy as np
import pip
import torch


# noinspection PyBroadException
def get_environment_details():
    try:
        lscpu = subprocess.check_output(['lscpu']).decode().split('\n')
        line_idx = [i for i, _ in enumerate(lscpu) if 'Model name' in lscpu[i]][0]
        cpu = lscpu[line_idx].split(':')[1].lstrip()
    except Exception:
        cpu = 'Unknown'

    try:
        lspci = subprocess.check_output(['lspci', '-vnn']).decode().split('\n')
        line_idx = [i for i, _ in enumerate(lspci) if torch.cuda.get_device_name(0).removeprefix('NVIDIA ') in lspci[i]][0]
        gpu = lspci[line_idx].split(': ')[1] + ' (' + lspci[line_idx + 1].split(': ')[1] + ')'
    except Exception:
        gpu = torch.cuda.get_device_name(0)

    try:
        nvidia = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                          '--format=csv,noheader']).decode().removesuffix('\n')
    except Exception:
        nvidia = 'Unknown'

    try:
        vbios = subprocess.check_output(['nvidia-smi', '--query-gpu=vbios_version',
                                         '--format=csv,noheader']).decode().removesuffix('\n')
    except Exception:
        vbios = 'Unknown'

    try:
        packages = {f'packages/{x.split("==")[0]}': x.split('==')[1]
                    for x in subprocess.check_output(['pip', 'list', '--format=freeze']).decode().split('\n')[:-1]}
    except Exception:
        packages = {}

    return {'job': os.environ.get('SLURM_JOBID', -1),
            'node': platform.node(),
            'cpu': cpu,
            'architecture': platform.processor(),
            'kernel': platform.release(),
            'glibc': platform.libc_ver()[1],
            'gpu': gpu,
            'nvidia': nvidia,
            'vbios': vbios,
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'python': platform.python_version(),
            'pip': pip.__version__,
            'numpy': np.__version__,
            'pytorch': torch.__version__,
            **packages}
