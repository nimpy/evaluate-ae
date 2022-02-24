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

sys.path.append('/scratch/cloned_repositories/sewar')
# from sewar import ssim as sewarssim
from sewar import msssim as msssim
from sewar import rmse_sw
from sewar import uqi
from sewar import ergas
from sewar import scc
# from sewar import rase
from sewar import sam
from sewar import vifp
from sewar import psnrb
# from sewar import d_lambda  # it makes no sense to use this -- it calculates Spectral Distortion Index
# the following 2 metrics also make no sense here, they have to do with panchromatic images and require a 3rd param:
# :param fused: high resolution fused image
# from sewar import d_s
# from sewar import qnr

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
    # diff_mse_cum = 0
    # diff_ssim_cum = 0
    # diff_psnr_cum = 0
    diff_msssim_cum = 0
    diff_are_cum = 0
    diff_ct_cum = 0
    diff_nrmse_eucl_cum = 0
    diff_nrmse_minmax_cum = 0
    diff_nrmse_mean_cum = 0
    diff_voi_oi_cum = 0
    diff_voi_io_cum = 0

    diff_rmse_sw_cum = 0
    diff_uqi_cum = 0
    diff_ergas_cum = 0
    diff_scc_cum = 0
    # diff_rase_cum = 0
    diff_sam_cum = 0
    diff_vifp_cum = 0
    diff_psnrb_cum = 0

    model.eval()

    for data_batch in test_dl:

        # data_batch = data_batch.cuda(non_blocking=True)  # uncomment for CUDA
        data_batch = Variable(data_batch)

        if variational:
            output_batch, _, _ = model(data_batch)
        else:
            output_batch = model(data_batch)

        # now convert the input and output batches to numpy
        data_batch = data_batch.cpu().numpy()
        output_batch = output_batch.detach().cpu().numpy()

        counter += data_batch.shape[0]

        for i in range(output_batch.shape[0]):
            # diff_mse = mse(data_batch[i], output_batch[i])
            # diff_mse_cum += diff_mse

            dr_max = max(data_batch[i].max(), output_batch[i].max())
            dr_min = min(data_batch[i].min(), output_batch[i].min())

            # diff_ssim = ssim(data_batch[i, 0], output_batch[i, 0], data_range=dr_max - dr_min)
            # diff_ssim_cum += diff_ssim

            # diff_psnr = psnr(data_batch[i, 0], output_batch[i, 0], data_range=dr_max - dr_min)
            # diff_psnr_cum += diff_psnr

            # for NRMSE, logging for different normalisations ('euclidean’, ‘min-max’, ‘mean')
            diff_nrmse_eucl = nrmse(data_batch[i], output_batch[i], normalization='euclidean')
            diff_nrmse_eucl_cum += diff_nrmse_eucl

            diff_nrmse_minmax = nrmse(data_batch[i], output_batch[i], normalization='min-max')
            diff_nrmse_minmax_cum += diff_nrmse_minmax

            diff_nrmse_mean = nrmse(data_batch[i], output_batch[i], normalization='mean')
            diff_nrmse_mean_cum += diff_nrmse_mean

        # transform the input and output batches into np.uint8
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

            # input and output images with shape of length 2 (matrices basically)
            input_image = data_batch[i][0]
            output_image = output_batch[i][0]

            diff_msssim = msssim(input_image, output_image).real
            # print('diff_msssim', diff_msssim)
            diff_msssim_cum += diff_msssim

            diff_rmse_sw = rmse_sw(input_image, output_image)[0]
            # print('diff_rmse_sw', diff_rmse_sw)
            diff_rmse_sw_cum += diff_rmse_sw

            diff_uqi = uqi(input_image, output_image)
            # print('diff_uqi', diff_uqi)
            diff_uqi_cum += diff_uqi

            diff_ergas = ergas(input_image, output_image)
            # print('diff_ergas', diff_ergas)
            diff_ergas_cum += diff_ergas

            diff_scc = scc(input_image, output_image)
            # print('diff_scc', diff_scc)
            diff_scc_cum += diff_scc

            # diff_rase = rase(input_image, output_image)
            # # print('diff_rase', diff_rase)
            # diff_rase_cum += diff_rase

            diff_sam = sam(input_image, output_image)
            # print('diff_sam', diff_sam)
            diff_sam_cum += diff_sam

            diff_vifp = vifp(input_image, output_image)
            # print('diff_vifp', diff_vifp)
            diff_vifp_cum += diff_vifp

            diff_psnrb = psnrb(input_image, output_image)
            # print('diff_psnrb', diff_psnrb)
            diff_psnrb_cum += diff_psnrb


    # diff_mse_average = diff_mse_cum / counter
    # diff_ssim_average = diff_ssim_cum / counter
    # diff_psnr_average = diff_psnr_cum / counter
    diff_msssim_average = diff_msssim_cum / counter

    diff_are_average = diff_are_cum / counter
    diff_ct_average = diff_ct_cum / counter

    diff_nrmse_eucl_average = diff_nrmse_eucl_cum / counter
    diff_nrmse_minmax_average = diff_nrmse_minmax_cum / counter
    diff_nrmse_mean_average = diff_nrmse_mean_cum / counter

    diff_voi_oi_average = diff_voi_oi_cum / counter
    diff_voi_io_average = diff_voi_io_cum / counter

    diff_rmse_sw_average = diff_rmse_sw_cum / counter
    diff_uqi_average = diff_uqi_cum / counter
    diff_ergas_average = diff_ergas_cum / counter
    diff_scc_average = diff_scc_cum / counter
    # diff_rase_average = diff_rase_cum / counter
    diff_sam_average = diff_sam_cum / counter
    diff_vifp_average = diff_vifp_cum / counter
    diff_psnrb_average = diff_psnrb_cum / counter

    # diff_mse_average, diff_ssim_average, diff_psnr_average, diff_msssim_average,
    # diff_rase_average
    return 0, 0, 0, diff_msssim_average, diff_are_average, \
            diff_ct_average, diff_nrmse_eucl_average, diff_nrmse_minmax_average, diff_nrmse_mean_average, \
            diff_voi_oi_average, diff_voi_io_average, \
            diff_rmse_sw_average, diff_uqi_average, diff_ergas_average, diff_scc_average, 0, \
            diff_sam_average, diff_vifp_average, diff_psnrb_average
