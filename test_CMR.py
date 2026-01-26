import glob
import itertools
import os
import monai
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

from argparse import ArgumentParser
from datetime import datetime
from datasets.dataloader import Dataset_Mapping_val
from utils.Functions import one_hot, dice
from model.reg_model_2D import VxmDense
from model.var_model import Variance_Decoder
from utils.layers import SpatialTransformer
from utils.losses import calculate_jacobian_metrics

ENC_NF = [128, 128, 128, 128]
DEC_NF = [256, 256, 256, 256, 256, 256, 256]
IMAGE_SIZE = (128, 128)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/data/yinsong/T1Map_Testing', help='test data path')
    parser.add_argument('--model_dir', type=str, default='../Model_CMR/', help='model directory')
    parser.add_argument('--reg_model', type=str, default='reg_model.pth', help='registration model filename')
    parser.add_argument('--var_model', type=str, default='var_model.pth', help='variance model filename')
    parser.add_argument('--proj_dim', default=32, type=int, help='projection head output dimension')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    return parser.parse_args()


def test(args):
    print("Testing started")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    transform_nearest = SpatialTransformer(IMAGE_SIZE, mode='nearest').to(device)

    names = sorted(glob.glob(args.datapath + '/P*/S*'))
    image_pair = []
    label_pair = []
    for name in names:
        fixed = sorted(glob.glob(name + '/s*_t0.nii.gz'))
        moving = sorted(glob.glob(name + '/s*_t?_warped.nii.gz'))
        fixed_label = glob.glob(name + '/T1map_label.nii.gz')
        fixed_label = fixed_label
        moving_label = sorted(glob.glob(name + '/s*_t?_warped_label.nii.gz'))
        image_pair += list(itertools.product(moving, fixed))
        label_pair += list(itertools.product(moving_label, fixed_label))

    test_dataset = Dataset_Mapping_val(image_pair, label_pair, norm=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    reg_model_path = os.path.join(args.model_dir, args.reg_model)
    var_model_path = os.path.join(args.model_dir, args.var_model)
    log_path = os.path.join(args.model_dir, 'Results_CMR.txt')

    with open(log_path, 'a') as log:
        log.write("Validation Results log: \n")

    reg_model = VxmDense(
        inshape=IMAGE_SIZE,
        nb_unet_features=[ENC_NF, DEC_NF],
        out_dim=args.proj_dim).to(device)
    print(f"Loading registration model: {reg_model_path}")
    reg_model.load_state_dict(torch.load(reg_model_path, map_location=device), strict=True)
    reg_model.eval()

    var_model = Variance_Decoder(
        inshape=IMAGE_SIZE,
        nb_unet_features=[ENC_NF, DEC_NF]).to(device)
    print(f"Loading variance model: {var_model_path}")
    var_model.load_state_dict(torch.load(var_model_path, map_location=device), strict=True)
    var_model.eval()

    dice_total = []
    grad_total = []
    jacobian_total = []
    hd95_total = []

    start_time = datetime.now()

    for data in test_loader:
        X = data[0].squeeze(1).to(device)
        Y = data[1].squeeze(1).to(device)
        X_label = data[2].squeeze(1).to(device)
        Y_label = data[3].squeeze(1).to(device)

        with torch.no_grad():
            _, _, F_X_Y, x_1, pred_his_1, trg_his_1 = reg_model(X, X, Y, Y, registration=True)
            _ = var_model(x_1, pred_his_1, trg_his_1)

            aligned_label = transform_nearest(X_label, F_X_Y)
            fixed_label_img = Y_label

            dice_score = dice(
                np.floor(aligned_label.cpu().numpy()[0, 0, :, :]),
                np.floor(fixed_label_img.cpu().numpy()[0, 0, :, :]))
            dice_total.append(dice_score)

            aligned_label_hot = one_hot(aligned_label, 4)
            fixed_label_img_hot = one_hot(fixed_label_img, 4)
            hd95_score = monai.metrics.compute_hausdorff_distance(aligned_label_hot, fixed_label_img_hot, percentile=95)[0, 1]
            hd95_total.append(hd95_score.cpu().numpy())

            F_X_Y_norm = F_X_Y.squeeze().cpu().numpy()
            fold_ratio, mag_grad_jac_de = calculate_jacobian_metrics(F_X_Y_norm)
            grad_total.append(mag_grad_jac_de)
            jacobian_total.append(fold_ratio)

    end_time = datetime.now()
    total_time = end_time - start_time

    with open(log_path, 'a') as log:
        log.write(f"{reg_model_path}:\n")
        log.write(f"(AVG) Dice: {np.mean(dice_total)}, Grad: {np.mean(grad_total)}, "
                  f"HD95: {np.mean(hd95_total)}, Fold: {np.mean(jacobian_total)}\n")
        log.write(f"(STD) Dice: {np.std(dice_total)}, Grad: {np.std(grad_total)}, "
                  f"HD95: {np.std(hd95_total)}, Fold: {np.std(jacobian_total)}\n")
        log.write(f"Total time + uncertainty: {total_time.total_seconds()}\n\n")

    pd.DataFrame(dice_total).to_csv(os.path.join(args.model_dir, 'dice.csv'), index=False)
    pd.DataFrame(grad_total).to_csv(os.path.join(args.model_dir, 'grad.csv'), index=False)
    pd.DataFrame(jacobian_total).to_csv(os.path.join(args.model_dir, 'jacobian.csv'), index=False)
    pd.DataFrame(hd95_total).to_csv(os.path.join(args.model_dir, 'hd95.csv'), index=False)

    print(f"Dice mean: {np.mean(dice_total)}")
    print(f"Grad mean: {np.mean(grad_total)}")
    print(f"Hausdorff mean: {np.mean(hd95_total)}")
    print(f"Jacobian mean: {np.mean(jacobian_total)}")


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    test(args)
    end_time = datetime.now()
    print(f"Total time: {(end_time - start_time).total_seconds():.2f}s")
