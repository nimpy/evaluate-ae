"""Unite different W+B sweeps into one W+B sweep for easier data manipulation and visualisation"""

import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import wandb
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--sweep_dir', default='/scratch/image_datasets/3_65x65/ready/weights',
                    help="Directory where the sweep runs are saved")
parser.add_argument('--saved_weights_dir', default='/home/niaki/Projects/local-img-descr-ae/weights_bak/sweep_all3',
                    help="Directory where the previously trained models are saved")

import evaluation
import models.ae as ae
import models.vae as vae


def load_sweep_csv(filepath):

    # I made this csv file by downloading csv files from different wandb sweeps and uniting them all into one csv file
    df = pd.read_csv(filepath)

    # replace . with _ in the column names so that pandas doesn't rename the columns
    new_columns = df.columns.values
    for i, column in enumerate(new_columns):
        new_column = column.replace('.', '_')
        new_columns[i] = new_column
    df.columns = new_columns
    return df


sweep_df = load_sweep_csv('sweep_results_times_dirs.csv')


def find_row_with_inputs(activation_fn, data_augm_level, loss_fn, vae_beta_norm):
    for row in sweep_df.itertuples():
        if row.activation_fn == activation_fn and row.data_augm_level == data_augm_level and \
                 row.loss_fn == loss_fn and row.vae_beta_norm == vae_beta_norm:
            print(row.Name)
            return row


def sweep_one_sweep_to_rule_them_all():

    args = parser.parse_args()

    use_wandb = True

    if use_wandb:
        wandb_run = wandb.init()

    # TODO delete when done with debugging
    # wandb.config.data_augm_level = 0
    # wandb.config.activation_fn = 'elu'
    # wandb.config.loss_fn = 'bce'
    # wandb.config.vae_beta_norm = 0.0001
    # wandb.config.learning_rate = 0.0001

    logging.info("\n\n****************** STARTING A NEW RUN ******************")
    logging.info('Data augmentation level: ' + str(wandb.config.data_augm_level))
    logging.info('Activation function    : ' + str(wandb.config.activation_fn))
    logging.info('Loss function          : ' + str(wandb.config.loss_fn))
    logging.info('Beta value (normalised): ' + str(wandb.config.vae_beta_norm))
    logging.info("")

    latent_size = 32
    batch_size = 32
    logging.info('Other params (that are not being swept)')
    logging.info('    Latent size:' + str(latent_size))
    logging.info('    Batch size :' + str(batch_size))
    logging.info("")

    wandb.config.variational = wandb.config.vae_beta_norm > 0.0000001
    wandb.config.latent_size = latent_size
    wandb.config.batch_size = batch_size
    wandb.config.num_workers = 4
    wandb.config.vae_or_ae = "vae" if wandb.config.vae_beta_norm > 0.0000001 else "ae"

    sweep_version = 'sweep__one_sweep_to_rule_them_all_v2022_1'  # TODO change in both files!!! (TODO make it a parameter)

    Path(os.path.join(args.sweep_dir, sweep_version)).mkdir(parents=True, exist_ok=True)

    # find and read the entry from the data frame
    row_df = find_row_with_inputs(wandb.config.activation_fn, wandb.config.data_augm_level, wandb.config.loss_fn, wandb.config.vae_beta_norm)

    # load the corresponding model
    model_dir_name = row_df.dir_name
    model_path = os.path.join(args.saved_weights_dir, model_dir_name, 'best.pth.tar')
    if model_dir_name.endswith('_vae'):
        model = vae.BetaVAE(32)
    else:
        model = ae.AE(32)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    # if torch.cuda.is_available():  # uncomment for CUDA
    #     model.cuda()

    mse, ssim, psnr, msssim, are, ct, nrmse_eucl, nrmse_minmax, nrmse_mean, voi_oi, voi_io, rmse_sw, uqi, ergas, scc, rase, sam, vifp, psnrb = evaluation.calculate_approximate_evaluation_metrics_on_test_set(model)  # mse, ssim, psnr
    print('--------- METRICS ---------')
    print(mse, ssim, psnr, msssim, are, ct, nrmse_eucl, nrmse_minmax, nrmse_mean, voi_oi, voi_io, rmse_sw, uqi, ergas, scc, rase, sam, vifp, psnrb)
    print()

    if use_wandb:
        wandb.log({"num_epochs": row_df.num_epochs, "variational": row_df.variational,
                   "hpatches_overall": row_df.hpatches_overall,
                   "matching_overall": row_df.matching_overall,
                   "retrieval_overall": row_df.retrieval_overall,
                   "verification_overall": row_df.verification_overall,
                   "mse": row_df.mse, "psnr": row_df.psnr, "ssim": row_df.ssim,
                   "are": are, "ct": ct, "nrmse_eucl": nrmse_eucl, "nrmse_minmax": nrmse_minmax, "nrmse_mean": nrmse_mean,
                   "voi_oi": voi_oi, "voi_io": voi_io, "rmse_sw": rmse_sw, "uqi": uqi, "ergas": ergas,
                   "scc": scc, "rase": rase, "sam": sam, "vifp": vifp, "psnrb": psnrb,
                   # "xxx": xxx, "xxx": xxx, "xxx": xxx, "xxx": xxx, "xxx": xxx,
                   # "xxx": xxx, "xxx": xxx, "xxx": xxx, "xxx": xxx, "xxx": xxx,
                   "loss": row_df.loss})

    if use_wandb:
        wandb_run.finish()
        
        
# if __name__ == '__main__':
#     sweep_one_sweep_to_rule_them_all()


    # if it's run like this (only for testing, then the following should be added right before the following line:
    # #    logging.info("\n\n****************** STARTING A NEW RUN ******************")

    # wandb.config.data_augm_level = 0
    # wandb.config.activation_fn = 'elu'
    # wandb.config.loss_fn = 'bce'
    # wandb.config.vae_beta_norm = 0.0001
    # wandb.config.learning_rate = 0.0001
