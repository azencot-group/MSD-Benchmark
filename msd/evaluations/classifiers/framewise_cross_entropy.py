import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class FramewiseCrossEntropyLoss(CrossEntropyLoss):
    """
    A variation of CrossEntropyLoss that applies the loss across each frame
    in a sequence, assuming the target is the same across all frames.

    This is useful when the label is static but the model outputs a prediction
    per frame, as in frame-level disentanglement settings.
    """

    def __init__(self, *args, **kwargs):
        """
        :param args: Arguments passed to standard CrossEntropyLoss.
        :param kwargs: Keyword arguments passed to standard CrossEntropyLoss.
        """
        super(FramewiseCrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        """
        Compute framewise cross-entropy by repeating the static target across
        the temporal dimension of the input.

        :param input_: Tensor of shape [B, T, C] — model logits per frame.
        :param target: Tensor of shape [B] — class index per sample.
        :return: Scalar loss averaged over all frames and samples.
        """
        t = input_.size(1)
        target = torch.stack([target] * t, dim=1).reshape(-1)
        input_ = input_.view(-1, input_.size(-1))
        return super(FramewiseCrossEntropyLoss, self).forward(input_, target)
