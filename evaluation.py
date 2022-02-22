from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import adapted_rand_error as are
from skimage.metrics import contingency_table as ct
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import variation_of_information as voi
# the following not part of skimage v0.17.2
# from skimage.metrics import hausdorff_distance as hd
# from skimage.metrics import hausdorff_pair as hp
# from skimage.metrics import normalized_mutual_information as nmi

import sys
sys.path.append('/scratch/cloned_repositories/pytorch-msssim')
from pytorch_msssim import msssim

import numpy as np
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
    diff_are_cum = 0
    diff_ct_cum = 0
    diff_nrmse_eucl_cum = 0
    diff_nrmse_minmax_cum = 0
    diff_nrmse_mean_cum = 0
    diff_voi_oi_cum = 0
    diff_voi_io_cum = 0

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

            # for NRMSE, logging for different normalisations ('euclidean’, ‘min-max’, ‘mean')
            diff_nrmse_eucl = nrmse(data_batch[i], output_batch[i], normalization='euclidean')
            diff_nrmse_eucl_cum += diff_nrmse_eucl

            diff_nrmse_minmax = nrmse(data_batch[i], output_batch[i], normalization='min-max')
            diff_nrmse_minmax_cum += diff_nrmse_minmax

            diff_nrmse_mean = nrmse(data_batch[i], output_batch[i], normalization='mean')
            diff_nrmse_mean_cum += diff_nrmse_mean

        data_batch = 255 * data_batch
        data_batch = data_batch.astype(np.uint8)
        output_batch = 255 * output_batch
        output_batch = output_batch.astype(np.uint8)

        for i in range(output_batch.shape[0]):

            diff_ct = ct(data_batch[i], output_batch[i])
            # print('diff_ct', diff_ct)
            # print('type', type(diff_ct))
            # print('--------')
            diff_ct_cum += 1#diff_ct  TODO

            # logging not only adapted random error but also its precision and recall
            diff_are, diff_are_prec, diff_are_rec = are(data_batch[i], output_batch[i])
            diff_are_cum += diff_are

            # with variation of information, logging both conditional entropies of image1|image0 and image0|image1
            diff_voi = voi(data_batch[i], output_batch[i])  # TODO 'create' another metric by summing them up
            diff_voi_oi = diff_voi[0]  # image1|image0
            diff_voi_io = diff_voi[1]  # image0|image1
            diff_voi_oi_cum += diff_voi_oi
            diff_voi_io_cum += diff_voi_io

    diff_mse_average = diff_mse_cum / counter
    diff_ssim_average = diff_ssim_cum / counter
    diff_psnr_average = diff_psnr_cum / counter
    diff_msssim_average = diff_msssim_cum.detach().cpu().numpy() / counter


    diff_are_average = diff_are_cum / counter
    diff_ct_average = diff_ct_cum / counter

    diff_nrmse_eucl_average = diff_nrmse_eucl_cum / counter
    diff_nrmse_minmax_average = diff_nrmse_minmax_cum / counter
    diff_nrmse_mean_average = diff_nrmse_mean_cum / counter

    diff_voi_oi_average = diff_voi_oi_cum / counter
    diff_voi_io_average = diff_voi_io_cum / counter


    return diff_mse_average, diff_ssim_average, diff_psnr_average, diff_msssim_average, diff_are_average, \
            diff_ct_average, diff_nrmse_eucl_average, diff_nrmse_minmax_average, diff_nrmse_mean_average, \
            diff_voi_oi_average, diff_voi_io_average
