import math
from math import exp
from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics
from torch import tensor, Tensor, nn
from torch.autograd import Variable


class Metrics:
    def __init__(self, MSE, RMSE, MAE, ABS_REL, LOG10, DELTA1, DELTA2, DELTA3, CADL):
        self.MSE = self.fix_number(MSE)
        self.RMSE = self.fix_number(RMSE)
        self.MAE = self.fix_number(MAE)
        self.ABS_REL = self.fix_number(ABS_REL)
        self.LOG10 = self.fix_number(LOG10)
        self.DELTA1 = self.fix_number(DELTA1)
        self.DELTA2 = self.fix_number(DELTA2)
        self.DELTA3 = self.fix_number(DELTA3)
        self.CADL = self.fix_number(CADL)

    def fix_number(self, num):
        if math.isnan(num):
            return 0  # Replace NaN with 0
        elif math.isinf(num):
            if num > 0:
                return float(99999999999)  # Replace positive infinity
            else:
                return float(99999999999)  # Replace negative infinity
        else:
            return num  # Return the number if it is neither NaN nor infinite


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class RootMeanAbsoluteError(torchmetrics.Metric):
    # https: // www.pytorchlightning.ai / blog / torchmetrics - pytorch - metrics - built - to - scale
    def __init__(self, **kwargs: Any, ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return torch.sqrt(self.sum_squared_error / self.n_observations)


class AbsoluteRelativeDifference(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("absolute_difference", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, tensor1, tensor2):
        absolute_difference = torch.abs(tensor1 - tensor2)
        denominator = torch.abs(tensor1)

        # Handle zero values
        zero_mask = (tensor1 == 0) & (tensor2 == 0)
        absolute_difference[zero_mask] = 0.0
        denominator[zero_mask] = 1.0  # Avoid division by zero

        # Handle negative values
        negative_mask = (tensor1 < 0) | (tensor2 < 0)
        absolute_difference[negative_mask] = 0.0
        denominator[negative_mask] = 1.0  # Avoid division by zero

        self.absolute_difference += torch.sum(absolute_difference)
        self.denominator += torch.sum(denominator)

    def compute(self):
        relative_difference = torch.zeros_like(self.denominator)
        non_zero_mask = self.denominator != 0
        relative_difference[non_zero_mask] = self.absolute_difference[non_zero_mask] / self.denominator[non_zero_mask]
        return relative_difference.mean()




class ConvAutoencoderDepthLoss(torchmetrics.Metric):
    # https: // www.pytorchlightning.ai / blog / torchmetrics - pytorch - metrics - built - to - scale
    def __init__(self):
        super().__init__()
        self.add_state("sum_error", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch_loss: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            batch_loss: Predictions from model for a given batch
        """

        self.sum_error += torch.sum(batch_loss)

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return self.sum_error


class Log10Metric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("num_examples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum_log10_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        log10_predictions = torch.log10(predictions)
        log10_targets = torch.log10(targets)

        # Handle edge cases where logarithm is undefined
        log10_predictions[torch.isnan(log10_predictions)] = 0.0
        log10_targets[torch.isnan(log10_targets)] = 0.0

        # Handle zero predictions and zero targets
        zero_mask = (predictions == 0) & (targets == 0)
        log10_predictions[zero_mask] = 0.0
        log10_targets[zero_mask] = 0.0

        # Handle negative predictions or targets
        negative_mask = (predictions < 0) | (targets < 0)
        log10_predictions[negative_mask] = 0.0
        log10_targets[negative_mask] = 0.0

        absolute_difference = torch.abs(log10_predictions - log10_targets)

        # Handle infinite predictions or targets
        infinite_mask = (~torch.isfinite(log10_predictions)) | (~torch.isfinite(log10_targets))
        absolute_difference[infinite_mask] = 0.0

        relative_difference = absolute_difference / (torch.abs(log10_targets) + 1e-8)
        relative_difference[torch.isnan(relative_difference)] = 0.0

        self.sum_log10_diff += torch.sum(relative_difference)
        self.num_examples += predictions.numel()

    def compute(self):
        return self.sum_log10_diff / self.num_examples


class Delta1(torchmetrics.Metric):
    # https://discuss.pytorch.org/t/what-does-1-25-1-25-1-25-delta-1-25-stand-for/174841
    def __init__(self, threshold=1.25):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate the absolute difference between predictions and targets
        abs_diff = torch.abs(preds - target)
        # Calculate the mask indicating which samples satisfy the Delta1 criterion
        mask = (abs_diff <= self.threshold)
        # Count the number of correct predictions
        correct_count = torch.sum(mask)
        # Update the state variables
        self.correct_count += correct_count
        self.total_count += target.numel()

    def compute(self):
        return self.correct_count.float() / self.total_count


class Delta2(torchmetrics.Metric):
    # https://discuss.pytorch.org/t/what-does-1-25-1-25-1-25-delta-1-25-stand-for/174841
    def __init__(self, threshold=1.25):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate the absolute difference between predictions and targets
        abs_diff = torch.abs(preds - target)
        # Calculate the mask indicating which samples satisfy the Delta1 criterion
        mask = (abs_diff <= math.pow(self.threshold, 2))
        # Count the number of correct predictions
        correct_count = torch.sum(mask)
        # Update the state variables
        self.correct_count += correct_count
        self.total_count += target.numel()

    def compute(self):
        return self.correct_count.float() / self.total_count


class Delta3(torchmetrics.Metric):
    # https://discuss.pytorch.org/t/what-does-1-25-1-25-1-25-delta-1-25-stand-for/174841
    def __init__(self, threshold=1.25):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate the absolute difference between predictions and targets
        abs_diff = torch.abs(preds - target)
        # Calculate the mask indicating which samples satisfy the Delta1 criterion
        mask = (abs_diff <= math.pow(self.threshold, 3))
        # Count the number of correct predictions
        correct_count = torch.sum(mask)
        # Update the state variables
        self.correct_count += correct_count
        self.total_count += target.numel()

    def compute(self):
        return self.correct_count.float() / self.total_count


# TODO Remove bellow once is tested

def lg10(x):
    return torch.div(torch.log(x), math.log(10))


def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z


def nValid(x):
    return torch.sum(torch.eq(x, x).float())


def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())


def getNanMask(x):
    return torch.ne(x, x)


def setNanToZero(input, target):
    target = target.movedim(2, -1)
    nanMask = getNanMask(target)
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target, nanMask, nValidElement


def evaluateError(output, target):
    errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
              'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    _output, _target, nanMask, nValidElement = setNanToZero(output, target)

    if (nValidElement.data.cpu().numpy() > 0):
        diffMatrix = torch.abs(_output - _target)

        errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement

        errors['MAE'] = torch.sum(diffMatrix) / nValidElement

        realMatrix = torch.div(diffMatrix, _target)
        realMatrix[nanMask] = 0
        errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement

        LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
        LG10Matrix[nanMask] = 0
        errors['LG10'] = torch.sum(LG10Matrix) / nValidElement

        yOverZ = torch.div(_output, _target)
        zOverY = torch.div(_target, _output)

        maxRatio = maxOfTwo(yOverZ, zOverY)

        errors['DELTA1'] = torch.sum(
            torch.le(maxRatio, 1.25).float()) / nValidElement
        errors['DELTA2'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
        errors['DELTA3'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

        errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
        errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
        errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
        errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
        errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
        errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
        errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

    return errors
