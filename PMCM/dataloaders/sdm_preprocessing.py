


import numpy as np
from glob import glob
from tqdm import tqdm
import h5py

import nibabel as nib
import pandas as pd

import pdb
import SimpleITK as sitk
from skimage import transform, measure
import os
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg




def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    # for b in range(out_shape[0]):  # batch size
    #     posmask = img_gt[b].astype(np.bool)
    #     if posmask.any():
    #         negmask = ~posmask
    #         posdis = distance(posmask)
    #         negdis = distance(negmask)
    #         boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
    #         sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
    #             np.max(posdis) - np.min(posdis))
    #         sdf[boundary == 1] = 0
    #         normalized_sdf[b] = sdf
    #         # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
    #         # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    # for b in range(out_shape[0]):  # batch size
    posmask = img_gt.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
        sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
            np.max(posdis) - np.min(posdis))
        sdf[boundary == 1] = 0
        normalized_sdf = sdf


    return normalized_sdf


if __name__ == "__main__":
    #LA
    base_dir = '/media/bspubuntu/1TBSSD/A_exp/dataset/LA_data'
    listt = glob('/media/bspubuntu/1TBSSD/A_exp/dataset/LA_data/*')

    # h5f = h5py.File(self._base_dir+"/LA_data/"+image_name+"/mri_norm2.h5", 'r')     #todo

    for item in tqdm(listt):
        name_image = str(item)

        item = item + "/mri_norm2.h5"
        h5f = h5py.File(item, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        gt_dis = compute_sdf(label, image.shape)
        name = item.replace('mri_norm2.h5', 'mri_dnorm2.h5')

        # f = h5py.File(name, 'w')

        # f.create_dataset('image', data=image, compression="gzip")
        # f.create_dataset('label', data=label, compression="gzip")
        # f.create_dataset('gt_dis', data=gt_dis, compression="gzip")
        # f.close()

    #Pancreas
    base_dir = '/media/bspubuntu/1TBSSD/A_exp/dataset/Pancreas_h5'
    listt = glob('/media/bspubuntu/1TBSSD/A_exp/dataset/Pancreas_h5/*')


    for item in tqdm(listt):
        name_image = str(item)

        h5f = h5py.File(item, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        gt_dis = compute_sdf(label, image.shape)
        oname = item.split('/')[-1]
        name = oname.replace('image', 'imaged')

        # f = h5py.File((base_dir+"/" + name), 'w')

        # f.create_dataset('image', data=image, compression="gzip")
        # f.create_dataset('label', data=label, compression="gzip")
        # f.create_dataset('gt_dis', data=gt_dis, compression="gzip")
        # f.close()