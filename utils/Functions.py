import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def generate_grid(inshape):
    vectors = [torch.arange(0, s) for s in inshape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)

    return grid


def generate_grid_unit(imgshape):
    """Generate a unit coordinate grid for given image shape.
    Args:
        imgshape: Tuple of (H, W, D) for 3D image dimensions

    Returns:
        Grid array with coordinates normalized to [-1, 1]
    """
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    """Convert unit flow [-1, 1] to pixel flow for 3D."""
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z - 1) / 2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y - 1) / 2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x - 1) / 2
    return flow


def transform_unit_flow_to_flow_2D(flow):
    """Convert unit flow [-1, 1] to pixel flow for 2D."""
    b, x, y, _ = flow.shape
    flow[:, :, 0] = flow[:, :, 0] * (y - 1) / 2
    flow[:, :, 1] = flow[:, :, 1] * (x - 1) / 2
    return flow


def transform_unit_flow_to_flow_cuda(flow):
    """Convert unit flow [-1, 1] to pixel flow for batched 3D on CUDA."""
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z - 1) / 2
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y - 1) / 2
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x - 1) / 2
    return flow


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute Soerensen-Dice coefficient between two masks.
    Args:
        mask_gt: Ground truth mask (boolean array)
        mask_pred: Predicted mask (boolean array)

    Returns:
        Dice coefficient as float. NaN if both masks are empty.
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def dice(fixed, moving_warped):
    """Compute mean Dice score across all classes.
    Args:
        fixed: Ground truth segmentation
        moving_warped: Warped segmentation to compare

    Returns:
        Mean Dice score across all present classes
    """
    dice_sum = 0
    count = 0
    unique_class = np.unique(fixed)
    for i in unique_class:
        if ((fixed == i).sum() == 0) or ((moving_warped == i).sum() == 0):
            continue
        dice_sum += compute_dice_coefficient((fixed == i), (moving_warped == i))
        count += 1
    return dice_sum / count if count > 0 else 0


def one_hot(seg_map, num_classes):
    """Convert segmentation map to one-hot encoding.
    Args:
        seg_map: Segmentation tensor of shape (B, 1, H, W, D)
        num_classes: Number of classes

    Returns:
        One-hot tensor of shape (B, num_classes, H, W, D)
    """
    seg_map = seg_map.squeeze(1).long()
    one_hot_map = F.one_hot(seg_map, num_classes)
    one_hot_map = one_hot_map.permute(0, 4, 1, 2, 3).float()
    return one_hot_map


def load_4D(name):
    """Load NIfTI image as 4D array (1, H, W, D)."""
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_4D_with_header(name):
    """Load NIfTI image with header and affine.
    """
    nii = nib.load(name)
    header, affine = nii.header, nii.affine
    X = nii.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X, header, affine


def load_5D(name):
    """Load NIfTI image as 5D array (1, 1, H, W, D)."""
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1, 1) + X.shape)
    return X


def imgnorm(img):
    """Normalize image to [0, 1] range (numpy)."""
    max_v = np.max(img)
    min_v = np.min(img)
    return (img - min_v) / (max_v - min_v)


def imgnorm_torch(img):
    """Normalize image to [0, 1] range (torch)."""
    max_v = torch.max(img)
    min_v = torch.min(img)
    return (img - min_v) / (max_v - min_v)


def save_img(img, savename, header=None, affine=None):
    """Save image as NIfTI file.
    """
    if affine is None:
        affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(img, affine, header=header)
    nib.save(new_img, savename)


def save_flow(flow, savename, header=None, affine=None):
    """Save flow field as NIfTI file.
    """
    save_img(flow, savename, header, affine)
