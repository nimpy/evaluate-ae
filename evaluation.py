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
    diff_mse_cum = 0
    diff_ssim_cum = 0
    diff_psnr_cum = 0
    diff_msssim_cum = 0

    model.eval()

    for data_batch in test_dl:

        # data_batch = data_batch.cuda(non_blocking=True)  # uncomment for CUDA
        data_batch = Variable(data_batch)

        if variational:
            output_batch, _, _ = model(data_batch)
        else:
            output_batch = model(data_batch)

        # calculating MS-SSIM before converting the input and output batches to numpy format...
        for input_image, output_image in zip(data_batch, output_batch):
            # reshape the images to add extra dimension because this is the expected input of pytorch_msssim
            # minus because I already negated msssim in its code  # TODO fix (install the library and import like that)
            diff_msssim = - msssim(torch.unsqueeze(input_image, 0), torch.unsqueeze(output_image, 0))
            diff_msssim_cum += diff_msssim

        # now convert the input and output batches to numpy
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

    diff_mse_average = diff_mse_cum / counter
    diff_ssim_average = diff_ssim_cum / counter
    diff_psnr_average = diff_psnr_cum / counter
    diff_msssim_average = diff_msssim_cum.detach().cpu().numpy() / counter

    return diff_mse_average, diff_ssim_average, diff_psnr_average, diff_msssim_average