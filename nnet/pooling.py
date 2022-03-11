from torch import nn
import torch

class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.
    It returns the concatenated mean and std of input tensor.
    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor = sp_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    """
    def __init__(self):
        super().__init__()
        # Small value for GaussNoise
        self.eps = 1e-5

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if lengths is None:
            mean = x.mean(dim=1)
            std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                mean.append(
                    torch.mean(x[snt_id, 1 : actual_size - 1, ...], dim=0)
                )
                std.append(
                    torch.std(x[snt_id, 1 : actual_size - 1, ...], dim=0)
                )
            mean = torch.stack(mean)
            std = torch.stack(std)
        gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
        gnoise = gnoise
        mean += gnoise
        std = std + self.eps
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(1)
        return pooled_stats

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.
        Arguments
        ---------
        shape_of_tensor : tensor
            It represents the size of tensor for generating Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)
        return gnoise