import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        #return nn.MSELoss(reduction='none')  # outputs loss for each batch sample
        #return MAPELoss(reduction='none', epsilon=0.01)  # outputs loss for each batch sample
        return MSEMAPELoss(reduction='none', epsilon=0.01, alpha=1000)

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)



# Reference:
# https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/regression/mape.py

class MAPELoss(nn.Module):
    """ Mean Absolute Percentage Error Loss
    """
    def __init__(self, reduction: str = 'mean', epsilon: float = 0.001):
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        abs_diff = torch.abs(y_pred - y_true)
        abs_per_error = abs_diff / torch.clamp(torch.abs(y_true), min=self.epsilon)

        if self.reduction == 'none':
            return abs_per_error
        elif self.reduction == 'mean':
            sum_abs_per_error = torch.sum(abs_per_error)
            num_obs = y_true.numel()
            return sum_abs_per_error / num_obs



class MSEMAPELoss(nn.Module):
    """ Mean Absolute Percentage Error Loss
    """
    def __init__(self, reduction: str = 'mean', epsilon: float = 0.001, alpha = 1000):
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        diff = y_pred - y_true
        abs_diff = torch.abs(diff)

        # absolute percentage error (calculated for each single element in minibatch)
        abs_per_error = abs_diff / torch.clamp(torch.abs(y_true), min=self.epsilon)

        # squre error
        square_error = torch.square(abs_diff)

        if self.reduction == 'none':
            return abs_per_error + self.alpha * square_error
        elif self.reduction == 'mean':
            sum_abs_per_error = torch.sum(abs_per_error)
            sum_square_error = torch.sum(square_error)
            num_obs = y_true.numel()
            return (sum_abs_per_error+self.alpha*sum_square_error) / num_obs