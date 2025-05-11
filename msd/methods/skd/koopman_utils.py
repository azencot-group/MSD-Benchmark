import numpy as np
import torch


def get_unique_num(D, I, static_number):
    """ This function gets a parameter for number of unique components. Unique is a component with imag part of 0 or
        couple of conjugate couple """
    i = 0
    for j in range(static_number):
        index = len(I) - i - 1
        val = D[I[index]]

        if val.imag == 0:
            i = i + 1
        else:
            i = i + 2

    return i


def get_sorted_indices(D, pick_type):
    """ Return the indexes of the eigenvalues (D) sorted by the metric chosen by an hyperparameter"""

    if pick_type == 'real':
        I = np.argsort(np.real(D))
    elif pick_type == 'norm':
        I = np.argsort(np.abs(D))
    elif pick_type == 'ball' or pick_type == 'space_ball':
        Dr = np.real(D)
        Db = np.sqrt((Dr - np.ones(len(Dr))) ** 2 + np.imag(D) ** 2)
        I = np.argsort(Db)
    else:
        raise Exception("no such method")

    return I


def static_dynamic_split(D, I, pick_type, static_size):
    """Return the eigenvalues indexes of the static and dynamic factors"""
    if pick_type == 'ball' or pick_type == 'space_ball':
        static_size = get_unique_num(D, I[::-1], static_size)
        Is, Id = I[:static_size], I[static_size:]
    else:
        static_size = get_unique_num(D, I, static_size)
        Id, Is = I[:-static_size], I[-static_size:]
    return Id, Is


def t_to_np(X):
    if X.dtype in [torch.float32, torch.float64]:
        X = X.detach().cpu().numpy()
    return X


def np_to_t(X, device='cuda'):
    if torch.cuda.is_available() is False:
        device = 'cpu'

    from numpy import dtype
    if X.dtype in [dtype('float32'), dtype('float64')]:
        X = torch.from_numpy(X.astype(np.float32)).to(device)
    return X
