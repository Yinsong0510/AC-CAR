import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from datasets.dataloader import Dataset_epoch_cam
from model.contrast_aug import GINGroupConv
from model.reg_model import VxmDense
from model.var_model import Variance_Decoder
from utils.losses import NCC_var, MSE_var, l2reg_loss_var, VarianceNLL, compute_SNR_from_variance_sim

ENC_NF = [32, 32, 32, 32]
DEC_NF = [64, 64, 64, 64, 64, 64, 64]
WARM_STEP = 40000

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--reglr', type=float, default=1e-4, help='registration model learning rate')
    parser.add_argument('--varlr', type=float, default=2e-5, help='variance model learning rate')
    parser.add_argument('--checkpoint', type=int, default=20000, help='frequency of saving models')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for beta-NLL loss')
    parser.add_argument('--proj_dim', default=16, type=int, help='projection head output dimension')
    parser.add_argument('--image_size', default=(192, 160, 144), type=tuple, help='image size')
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='MSE loss weight')
    parser.add_argument('--lambda_ncc', type=float, default=0.8, help='NCC loss weight')
    parser.add_argument('--lambda_reg', type=float, default=0.15, help='regularization loss weight')
    parser.add_argument('--lambda_clr', type=float, default=0.1, help='CLR loss weight')
    parser.add_argument('--datapath', type=str, default='../../IXI_final/IXI-train', help='training data path')
    parser.add_argument('--model_dir', type=str, default='../Model_IXI', help='model save directory')
    parser.add_argument('--log_dir', type=str, default='../Log_IXI', help='tensorboard log directory')
    return parser.parse_args()


def train(args):
    print("Training started")
    device = torch.device('cuda:2')
    torch.cuda.set_device(device)

    reg_model = VxmDense(
        inshape=args.image_size,
        nb_unet_features=[ENC_NF, DEC_NF],
        out_dim=args.proj_dim).to(device)
    reg_model.train()

    var_model = Variance_Decoder(
        inshape=args.image_size,
        nb_unet_features=[ENC_NF, DEC_NF]).to(device)
    var_model.train()

    mse_loss = MSE_var().loss
    ncc_loss = NCC_var().loss
    var_loss_fn = VarianceNLL(beta=args.beta)
    contra_loss = torch.nn.MSELoss()

    contrast_aug = GINGroupConv().to(device)

    image_paths = sorted(glob.glob(os.path.join(args.datapath, 'IXI-T1/IXI*.nii.gz')))
    train_dataset = Dataset_epoch_cam(image_paths, norm=True)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    optimizer_reg = torch.optim.Adam(reg_model.parameters(), lr=args.reglr, weight_decay=1e-5)
    optimizer_var = torch.optim.Adam(var_model.parameters(), lr=args.varlr, weight_decay=1e-5)
    scheduler_reg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_reg, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    os.makedirs(args.model_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    step = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        for X, Y in train_loader:
            if step < WARM_STEP:
                flag_motion, flag_variance = True, False
            elif step < 2 * WARM_STEP:
                flag_motion, flag_variance = False, True
            else:
                flag_motion, flag_variance = True, True

            X = X.squeeze(-1).to(device).float()
            Y = Y.squeeze(-1).to(device).float()
            X_aug_1 = contrast_aug(X)
            Y_aug_1 = contrast_aug(Y)

            if flag_motion:
                X_aug_2 = contrast_aug(X)
                Y_aug_2 = contrast_aug(Y)

            mask_bgd = (X != 0) + (Y != 0)

            y_pred_1, y_pa_1, flow_1, sv_1, tv_1, sd_1, td_1 = reg_model(X_aug_1, X, Y_aug_1, Y, registration=True)
            y_pa_1_detached = y_pa_1.detach()
            x_1, pred_his_1, trg_his_1, _, _, _, _ = reg_model(y_pa_1_detached, y_pa_1_detached, Y_aug_1, Y_aug_1, registration=False)
            del y_pa_1

            if flag_variance:
                det_pred_his_1 = [t.clone().detach() for t in pred_his_1]
                det_trg_his_1 = [t.clone().detach() for t in trg_his_1]
                clone_x_1 = x_1.clone().detach()

            if flag_motion:
                _, _, _, sv_2, tv_2, sd_2, td_2 = reg_model(X_aug_2, X, Y_aug_2, Y, registration=False)
                with torch.no_grad():
                    log_var_1 = var_model(x_1.detach(), pred_his_1, trg_his_1)
                    var_1 = compute_SNR_from_variance_sim(Y, log_var_1, mask_bgd)

                loss_mse = mse_loss(Y, y_pred_1, var_1.detach()) * args.lambda_mse
                loss_ncc = ncc_loss(Y, y_pred_1, var_1.detach()) * args.lambda_ncc
                loss_reg = l2reg_loss_var(flow_1, var_1.detach()) * args.lambda_reg

                loss_vec = contra_loss(sv_1, sv_2) + contra_loss(tv_1, tv_2)
                loss_dense = contra_loss(sd_1, sd_2) + contra_loss(td_1, td_2)
                loss_clr = (loss_vec + loss_dense) * args.lambda_clr

                loss = loss_mse + loss_ncc + loss_reg + loss_clr

                optimizer_reg.zero_grad()
                loss.backward()
                optimizer_reg.step()
                scheduler_reg.step()

                sys.stdout.write(
                    f"\rstep {step} - mse: {loss_mse.item():.4f} - ncc: {loss_ncc.item():.4f} - "
                    f"reg: {loss_reg.item():.4f} - clr: {loss_clr.item():.4f}")
                sys.stdout.flush()

                writer.add_scalar('Loss/total', loss.item(), step)
                writer.add_scalar('Loss/mse', loss_mse.item(), step)
                writer.add_scalar('Loss/ncc', loss_ncc.item(), step)
                writer.add_scalar('Loss/reg', loss_reg.item(), step)
                writer.add_scalar('Loss/clr', loss_clr.item(), step)

                del log_var_1, var_1, flow_1, X_aug_2, Y_aug_2, sv_2, tv_2, sd_2, td_2
                del loss, loss_mse, loss_ncc, loss_reg, loss_vec, loss_dense, loss_clr
                if flag_variance:
                    del x_1, pred_his_1, trg_his_1
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if flag_variance:
                log_var = var_model(clone_x_1, det_pred_his_1, det_trg_his_1)
                loss_var = var_loss_fn.loss(y_pred_1.detach(), Y, log_var, mask_bgd)

                optimizer_var.zero_grad()
                loss_var.backward()
                optimizer_var.step()

                log_var_mean = torch.mean(log_var).item()
                sys.stdout.write(f"\rstep {step} - var: {loss_var.item():.4f} - log_var_mean: {log_var_mean:.4f}")
                sys.stdout.flush()

                writer.add_scalar('Loss/var', loss_var.item(), step)
                writer.add_scalar('Loss/log_var_mean', log_var_mean, step)

                del log_var, loss_var, clone_x_1, det_pred_his_1, det_trg_his_1
                if not flag_motion:
                    del x_1, pred_his_1, trg_his_1
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if flag_motion or flag_variance:
                try:
                    del y_pred_1, y_pa_1_detached
                    if flag_motion:
                        del sv_1, tv_1, sd_1, td_1
                except (NameError, UnboundLocalError):
                    pass

            step += 1

            if step % args.checkpoint == 0:
                var_path = os.path.join(args.model_dir, f'var_model_{step}.pth')
                reg_path = os.path.join(args.model_dir, f'reg_model_{step}.pth')
                torch.save(var_model.state_dict(), var_path)
                torch.save(reg_model.state_dict(), reg_path)
                print(f"\nSaved checkpoints: {var_path}, {reg_path}")
                torch.cuda.empty_cache()

    writer.close()
    print("\nTraining completed")


if __name__ == "__main__":
    args = parse_args()
    start_time = datetime.now()
    train(args)
    end_time = datetime.now()
    print(f"Total time: {(end_time - start_time).total_seconds():.2f}s")
