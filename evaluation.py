from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

import sys
sys.path.append('/scratch/cloned_repositories/pytorch-msssim')
from pytorch_msssim import msssim

import torch
from torch.autograd import Variable

import utilities
# import models.ae as ae
import models.vae as vae
import data_loader


def calculate_approximate_evaluation_metrics_on_test_set(model):

    params = utilities.Params('models/params.json')
    params.cuda = False  # torch.cuda.is_available()  # uncomment for CUDA

    variational = isinstance(model, vae.BetaVAE)

    dataloaders = data_loader.fetch_dataloader(['test'], '/scratch/image_datasets/3_65x65/ready', params, batch_size=32)
    test_dl = dataloaders['test']

    counter = 0
    counter_msssim = 0
    diff_mse_cum = 0
    diff_ssim_cum = 0
    diff_psnr_cum = 0
    diff_msssim_cum1 = 0
    diff_msssim_cum2 = 0

    model.eval()

    for data_batch in test_dl:

        # data_batch = data_batch.cuda(non_blocking=True)  # uncomment for CUDA
        data_batch = Variable(data_batch)

        if variational:
            output_batch, _, _ = model(data_batch)
        else:
            output_batch = model(data_batch)

        diff_msssim1 = - msssim(data_batch, output_batch)  # minus because I already inverted it... # TODO fix
        diff_msssim_cum1 += diff_msssim1
        # print(diff_msssim_cum)
        counter_msssim += 1
        print(data_batch.shape)

        for input_image, output_image in zip(data_batch, output_batch):
            diff_msssim2 = - msssim(torch.unsqueeze(input_image, 0), torch.unsqueeze(output_image, 0))
            diff_msssim_cum2 += diff_msssim2


        data_batch = data_batch.cpu().numpy()
        output_batch = output_batch.detach().cpu().numpy()

        counter += data_batch.shape[0]

        for i in range(output_batch.shape[0]):
            diff_mse = mse(data_batch[i], output_batch[i])
            diff_mse_cum += diff_mse

            dr_max = max(data_batch[i].max(), output_batch[i].max())
            dr_min = min(data_batch[i].min(), output_batch[i].min())

            diff_ssim = ssim(data_batch[i, 0], output_batch[i, 0], data_range=dr_max - dr_min)
            diff_ssim_cum += diff_ssim

            diff_psnr = psnr(data_batch[i, 0], output_batch[i, 0], data_range=dr_max - dr_min)
            diff_psnr_cum += diff_psnr

            # diff_msssim = - msssim(data_batch[i], output_batch[i])  # minus because I already inverted it... # TODO fix
            # diff_msssim_cum += diff_msssim

    diff_mse_average = diff_mse_cum / counter
    diff_ssim_average = diff_ssim_cum / counter
    diff_psnr_average = diff_psnr_cum / counter
    diff_msssim_average1 = diff_msssim_cum1 / counter_msssim
    diff_msssim_average2 = diff_msssim_cum2 / counter
    # these should be the same but they are not exactly the same because the last batch has 20 images instead of 32
    print(diff_msssim_average1, '=================', diff_msssim_average2)

    return diff_mse_average, diff_ssim_average, diff_psnr_average, diff_msssim_average1