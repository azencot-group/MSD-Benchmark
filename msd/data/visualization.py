import textwrap
from typing import List, Optional, Any

import imageio
import matplotlib.pyplot as plt
import numpy as np


def plot_image(
    sample: np.ndarray,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    out_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a single image (CHW format) with optional saving and display.

    :param sample: Image array of shape (C, H, W).
    :param fig: Optional matplotlib Figure to use.
    :param ax: Optional matplotlib Axes to use.
    :param out_path: Optional path to save the figure.
    :param show: Whether to display the image using plt.show().
    :return: The figure object used.
    """
    sample = sample.transpose(1, 2, 0)
    if sample.max() <= 1:
        sample = sample * 255
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(np.clip(sample.squeeze(), 0, 255).astype('uint8'))
    ax.axis('off')
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    return fig


def plot_sequence(
    sample: np.ndarray,
    suptitle: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[List[plt.Axes]] = None,
    out_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a sequence of images in a single row.

    :param sample: A sequence of images of shape (T, C, H, W).
    :param suptitle: Optional title above the entire sequence.
    :param fig: Optional matplotlib Figure to use.
    :param ax: Optional list of Axes to draw on.
    :param out_path: Optional path to save the figure.
    :param show: Whether to display the figure.
    :return: The figure object used.
    """
    sample = sample.transpose(0, 2, 3, 1)
    if sample.max() <= 1:
        sample = sample * 255
    T = sample.shape[0]
    if ax is None:
        fig, ax = plt.subplots(1, T, figsize=(T, 1))
    fig.subplots_adjust(wspace=0)
    for i, im in enumerate(sample):
        ax[i].imshow(np.clip(im.squeeze(), 0, 255).astype('uint8'))
        ax[i].axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    return fig


def plot_sequences(
    samples: List[np.ndarray],
    titles: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    axs: Optional[Any] = None,
    out_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a list of image sequences, one per row.

    :param samples: List of sequences (each of shape T x C x H x W).
    :param titles: Optional list of titles per row.
    :param suptitle: Optional super title for the entire figure.
    :param fig: Optional matplotlib Figure.
    :param axs: Optional array of Axes.
    :param out_path: Optional path to save the figure.
    :param show: Whether to display the plot.
    :return: The figure object used.
    """
    if titles is None:
        titles = [''] * len(samples)
    N = len(samples)
    T = samples[0].data.shape[0]
    max_title_length = 3
    left_margin = 0.1 + 0.02 * max_title_length

    if axs is None:
        fig, axs = plt.subplots(N, T, figsize=(T, N))
    fig.subplots_adjust(left=left_margin, wspace=0)

    for i, sample in enumerate(samples):
        plot_sequence(sample, suptitle=suptitle, fig=fig, ax=axs[i], show=False)
        wrapped_title = textwrap.fill(titles[i], 20)
        axs[i, 0].text(-0.5, 0.5, wrapped_title, fontsize=12, rotation=0, ha='right', va='center',
                       transform=axs[i, 0].transAxes)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    return fig


def compare_sequences(
    samples1: List[np.ndarray],
    titles1: List[str],
    samples2: List[np.ndarray],
    titles2: List[str],
    suptitle: Optional[str] = None,
    out_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot two sets of sequences side-by-side for visual comparison.

    :param samples1: First set of sequences (list of T x C x H x W arrays).
    :param titles1: Titles for the first set.
    :param samples2: Second set of sequences (list of T x C x H x W arrays).
    :param titles2: Titles for the second set.
    :param suptitle: Optional super title.
    :param out_path: Optional path to save the plot.
    :param show: Whether to display the figure.
    :return: The resulting figure.
    """
    N = len(samples1)
    T = samples1[0].shape[0]
    k = 4
    fig, axs = plt.subplots(N, 2 * T + k, figsize=(2 * T + k, N))
    fig.subplots_adjust(wspace=0)
    for i in range(N):
        for j in range(k):
            axs[i, T + j].axis('off')
    plot_sequences(samples1, titles1, fig=fig, axs=axs[:, :T], show=False)
    plot_sequences(samples2, titles2, fig=fig, axs=axs[:, T+k:], show=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    return fig


def create_video(
    sample: np.ndarray,
    out_path: str,
    fps: int = 3
) -> None:
    """
    Create a .mp4 video from a sequence of images.

    :param sample: A sequence of images (T x C x H x W).
    :param out_path: Path to save the output video file.
    :param fps: Frames per second for the output video. Optional, default is 3.
    """
    sample = sample.transpose(0, 2, 3, 1)
    sample = (sample / sample.max() * 255).astype(np.uint8)

    imageio.mimsave(out_path, sample, fps=fps)

