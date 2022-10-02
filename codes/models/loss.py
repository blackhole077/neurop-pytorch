import torch
import torch.nn as nn


class TVLoss(nn.Module):
    """A class encapsulating the Total Variation (TV) loss function.

    What is it
    ----------
    Total Variation, defined as Remi Flamary here (https://remi.flamary.com/demos/proxtv.html)
    can be considered as a regularization on the noise of the image. While there are 3 different types of regularization,
    the one that is implemented here is the squared L2 Total Variation Loss (L2 refers to the Euclidean distance).
    
    """

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x: torch.Tensor):
        batch_size, _, h_x, w_x = x.size()
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t: torch.Tensor) -> int:
        return t.size()[1] * t.size()[2] * t.size()[3]


class CosineLoss(nn.Module):
    """A class encapsulation of the Cosine loss function.

    This calculates the cosine similarity on a per-pixel basis for a batch of images.
    For this to work, it assumes that the inputs are both tensors where the first one, termed x, has the following information:
        - batch_size: The number of images in the current batch for training/validation/testing
        - image_channels: The number of channels in the image (e.g., RGB images are expected to have 3 channels)
        - image_height: The height of all images in the batch, in pixels
        - image_width: The width of all images in the batch, in pixels

    Presumably, the ground truth examples, termed y, should have similar information and thus an identical shape to the input x.
    """

    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # This assumes that x is a 4D Tensor whose shape corresponds to the following values
        batch_size, image_channels, image_height, image_width = x.size()
        # Permute just swaps the order of axes around. In this case, it could be assumed that the order becomes (batch_size, image_height, image_width, image_channels)
        # View works like numpy reshape (i.e., it changes the chape of the tensor to the specified dimensions where -1 indicates that particular dimension should be figured out automatically)
        x = x.permute(0, 2, 3, 1).view(-1, image_channels)
        y = y.permute(0, 2, 3, 1).view(-1, image_channels)
        # Generate the per pixel(?) loss value for the entire batch
        cosine_loss_value = 1.0 - self.cos(x, y).sum() / (
            1.0 * batch_size * image_height * image_width
        )
        return cosine_loss_value
