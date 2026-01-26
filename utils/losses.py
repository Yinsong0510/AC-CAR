import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk


class NCC:
    """Local (over window) normalized cross correlation loss."""

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):
        ndims = len(list(y_true.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        win = [9] * ndims if self.win is None else self.win
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)
        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride, padding = (1,), (pad_no,)
        elif ndims == 2:
            stride, padding = (1, 1), (pad_no, pad_no)
        else:
            stride, padding = (1, 1, 1), (pad_no, pad_no, pad_no)

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = y_true * y_true
        J2 = y_pred * y_pred
        IJ = y_true * y_pred

        I_sum = conv_fn(y_true, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(y_pred, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)


class NCC_var:
    """Local normalized cross correlation loss with variance weighting."""

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred, var):
        ndims = len(list(y_true.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        win = [9] * ndims if self.win is None else self.win
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)
        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride, padding = (1,), (pad_no,)
        elif ndims == 2:
            stride, padding = (1, 1), (pad_no, pad_no)
        else:
            stride, padding = (1, 1, 1), (pad_no, pad_no, pad_no)

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = y_true * y_true
        J2 = y_pred * y_pred
        IJ = y_true * y_pred

        I_sum = conv_fn(y_true, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(y_pred, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        return -torch.mean(cc)


class MSE:
    """Mean squared error loss."""

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class MSE_var:
    """Mean squared error loss with variance weighting."""

    def loss(self, y_true, y_pred, var):
        return torch.mean(((y_true - y_pred) ** 2) * var)


class MSE_mask:
    """Mean squared error loss with mask."""

    def loss(self, y_true, y_pred, mask):
        return torch.sum((y_true - y_pred) ** 2 * mask) / torch.sum(mask)


class DiceLoss(nn.Module):
    """Dice loss with optional one-hot encoding."""

    def __init__(self, num_class=4, if_onehot=False):
        super().__init__()
        self.num_class = num_class
        self.if_onehot = if_onehot

    def forward(self, y_pred, y_true):
        if self.if_onehot:
            y_true = F.one_hot(y_true, num_classes=self.num_class)
            y_true = torch.squeeze(y_true, 1)
            y_true = y_true.permute(0, 3, 1, 2).contiguous()

        intersection = (y_pred * y_true).sum(dim=[2, 3])
        union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3])
        dsc = (2. * intersection) / (union + 1e-5)
        return 1 - torch.mean(dsc)


class MIND_loss(nn.Module):
    """MIND-SSC (Modality Independent Neighbourhood Descriptor) loss."""

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        kernel_size = radius * 2 + 1
        device = img.device

        six_neighbourhood = torch.tensor([
            [0, 1, 1], [1, 1, 0], [1, 0, 1],
            [1, 1, 2], [2, 1, 1], [1, 2, 1]
        ]).long()

        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]

        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1

        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        ssd = F.avg_pool3d(
            rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) -
                   F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
            kernel_size, stride=1
        )

        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, mind_var.mean().item() * 0.001, mind_var.mean().item() * 1000)
        mind = torch.exp(-mind / mind_var)

        # Permute to match C++ code ordering
        mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)


class MIND_loss_var(nn.Module):
    """MIND-SSC loss with variance weighting."""

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        kernel_size = radius * 2 + 1
        device = img.device

        six_neighbourhood = torch.tensor([
            [0, 1, 1], [1, 1, 0], [1, 0, 1],
            [1, 1, 2], [2, 1, 1], [1, 2, 1]
        ]).long()

        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]

        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1

        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        ssd = F.avg_pool3d(
            rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) -
                   F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
            kernel_size, stride=1
        )

        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, mind_var.mean().item() * 0.001, mind_var.mean().item() * 1000)
        mind = torch.exp(-mind / mind_var)

        mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
        return mind

    def forward(self, y_pred, y_true, var):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)


class Grad3d(nn.Module):
    """N-D gradient loss for deformation field smoothness."""

    def __init__(self, penalty='l2', loss_mult=None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy, dx, dz = dy * dy, dx * dx, dz * dz

        grad = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


def l2reg_loss(u):
    """L2 regularization loss using finite differences."""
    derives = [finite_diff(u, dim=i) for i in range(u.size(1))]
    return torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()


def l2reg_loss_var(u, var):
    """L2 regularization loss with variance weighting."""
    derives = [finite_diff(u, dim=i) for i in range(u.size(1))]
    return torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()


def finite_diff(x, dim, mode="forward", boundary="Neumann"):
    """Compute finite differences along a dimension.

    Args:
        x: Input tensor of shape (N, ndim, *sizes)
        dim: Dimension to compute difference along
        mode: 'forward' or 'backward'
        boundary: 'Neumann' (replicate) or 'Dirichlet' (zero)
    """
    assert isinstance(x, torch.Tensor)
    ndim = x.ndim - 2
    sizes = x.shape[2:]

    if mode == "central":
        raise NotImplementedError("Central difference mode not implemented")

    # Configure padding
    paddings = [[0, 0] for _ in range(ndim)]
    if mode == "forward":
        paddings[dim][1] = 1
    elif mode == "backward":
        paddings[dim][0] = 1
    else:
        raise ValueError(f'Mode {mode} not recognised')

    # Flatten padding list (PyTorch uses last -> first dim order)
    paddings.reverse()
    paddings = [p for ppair in paddings for p in ppair]

    # Apply padding
    pad_mode = 'replicate' if boundary == "Neumann" else 'constant'
    if boundary not in ["Neumann", "Dirichlet"]:
        raise ValueError("Boundary condition not recognised.")
    x_pad = F.pad(x, paddings, mode=pad_mode)

    # Compute difference
    x_diff = (x_pad.index_select(dim + 2, torch.arange(1, sizes[dim] + 1).to(device=x.device)) -
              x_pad.index_select(dim + 2, torch.arange(0, sizes[dim]).to(device=x.device)))
    return x_diff


class NLLloss:
    """Negative log-likelihood loss with optional beta weighting.

    References:
        NLL: https://proceedings.neurips.cc/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
        beta-NLL: https://github.com/martius-lab/beta-nll
    """

    def __init__(self, beta=0.0, flag_laplace=False):
        self.beta = beta
        self.flag_laplace = flag_laplace

    def loss(self, y_true, y_pred, logsigma_image, mask_bgd):
        assert y_true.size() == y_pred.size() == logsigma_image.size()

        y_true = torch.mul(y_true.detach(), mask_bgd)
        y_pred = torch.mul(y_pred.detach(), mask_bgd)
        logsigma_image = torch.mul(logsigma_image, mask_bgd)

        if self.flag_laplace:
            error = 0.5 * torch.mul(torch.abs(y_true - y_pred), torch.exp(-logsigma_image)) + 0.5 * logsigma_image
        else:
            error = 0.5 * torch.mul(torch.square(y_true - y_pred), torch.exp(-logsigma_image)) + 0.5 * logsigma_image

        if self.beta > 0:
            error = error * (torch.exp(logsigma_image.detach()) ** self.beta)

        return torch.mean(error)


# Alias for backward compatibility
VarianceNLL = NLLloss


def compute_SNR_from_variance(y_true, logsigma_image, mask_bgd, gamma=0.5):
    """Compute signal-to-noise ratio from variance estimate."""
    assert y_true.size() == logsigma_image.size()

    y_true = torch.mul(y_true, mask_bgd)
    logsigma_image = torch.mul(logsigma_image, mask_bgd)

    rel_snr = (torch.square(y_true) / torch.exp(logsigma_image)) ** gamma
    rel_snr = (rel_snr - torch.min(rel_snr)) / (torch.max(rel_snr) - torch.min(rel_snr))
    return torch.sigmoid(rel_snr)


def compute_SNR_from_variance_sim(y_true, logsigma_image, mask_bgd, gamma=0.5):
    """Compute simplified SNR from variance estimate."""
    assert y_true.size() == logsigma_image.size()

    y_true = torch.mul(y_true, mask_bgd)
    logsigma_image = torch.mul(logsigma_image, mask_bgd)

    rel_snr = (1 / torch.exp(logsigma_image)) ** gamma
    rel_snr = (rel_snr - torch.min(rel_snr)) / (torch.max(rel_snr) - torch.min(rel_snr))
    return rel_snr


def intersection_mask(mask1, mask2):
    """Compute intersection of two binary masks."""
    return (mask1.bool() & mask2.bool()).float()


def sparsification_plot(prediction, ground_truth, uncertainty, save_path="../Results/sparsification_plot.png"):
    """Generate sparsification plot for uncertainty evaluation.

    Args:
        prediction: Predicted values
        ground_truth: Ground truth values
        uncertainty: Uncertainty estimates
        save_path: Path to save the plot
    """
    from matplotlib import pyplot as plt

    prediction_flat = prediction.reshape(-1)
    ground_truth_flat = ground_truth.reshape(-1)
    uncertainty_flat = uncertainty.reshape(-1)

    initial_error = ((prediction_flat - ground_truth_flat) ** 2).mean().item()
    print(f"Initial error: {initial_error}")

    sorted_indices = torch.argsort(uncertainty_flat, descending=True)

    errors = []
    remaining_pixels = []
    num_pixels = len(prediction_flat)

    for i in range(0, num_pixels, num_pixels // 100):
        keep_indices = sorted_indices[i:]
        error = ((prediction_flat[keep_indices] - ground_truth_flat[keep_indices]) ** 2).mean()
        errors.append(error.item())
        remaining_pixels.append(1 - i / num_pixels)

    plt.figure(figsize=(7, 7))
    plt.plot(remaining_pixels, errors, marker='.', linewidth=3)
    plt.grid(c='g', linestyle='--', linewidth=2, alpha=0.5, rasterized=True)
    plt.xlabel("Proportion of pixels retained", fontsize=22)
    plt.ylabel("Sparsification error for MSE", fontsize=18)
    plt.yticks(size=18)
    plt.xticks(fontsize=18)
    plt.gca().invert_xaxis()
    plt.savefig(save_path)
    plt.close()


def calculate_jacobian_metrics(disp):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """
    disp_n = np.moveaxis(disp, 0, -1)  # (*sizes, ndim)
    jac_det_n = calculate_jacobian_det(disp_n)
    folding_ratio = (jac_det_n < 0).sum() / np.prod(jac_det_n.shape)
    mag_grad_jac_det = np.abs(np.gradient(jac_det_n)).mean()
    return folding_ratio, mag_grad_jac_det


def calculate_jacobian_det(disp):
    """
    Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)

    Args:
        disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field

    Returns:
        jac_det: (numpy.adarray, shape (*sizes) Point-wise Jacobian determinant
    """
    disp_img = sitk.GetImageFromArray(disp, isVector=True)
    jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
    jac_det = sitk.GetArrayFromImage(jac_det_img)
    return jac_det