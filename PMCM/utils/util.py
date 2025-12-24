import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torchvision.utils import make_grid
import networks

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_grid(data, name, channel, grid, RGB = False):

    if not RGB:
        # class 1
        assert len(data.shape) == 5
        # BCHWD ->　HWD ->　1HWD -> D1HW -> D3HW
        image = data[0, channel, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_v_b = make_grid(image, 5, normalize=False)
        grid[name] = grid_v_b
    else:
        assert len(data.shape) == 4
        # CHWD -> 
        image = data[:, :, 20:61:10, :].transpose(2, 3, 0, 1)
        # image = data[0, :, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        grid_v_b = make_grid(torch.from_numpy(image), 5, normalize=False)
        grid[name] = grid_v_b

    return grid


def plot_overlap(pred, gt):
    assert len(pred.shape) == 3
    overlap_image = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
    
    # True Positive (TP): 預測和 ground truth 同時為 1
    tp = (pred == 1) & (gt == 1)
    # False Negative (FN): 預測為 0，ground truth 為 1 -> 標為紅色
    fn = (pred == 0) & (gt == 1)
    # False Positive (FP): 預測為 1，ground truth 為 0 -> 標為綠色
    fp = (pred == 1) & (gt == 0)

    # 設置 TP 為白色
    overlap_image[tp] = [255, 255, 255]  # 白色
    # 設置 FN 為紅色
    overlap_image[fn] = [255, 0, 0]      # 紅色
    # 設置 FP 為綠色
    overlap_image[fp] = [0, 255, 0]      # 綠色

    return overlap_image

def plot_KDE(decoder1_output, decoder2_output, save_path, iter_num, channel = 0, ul = False):

    if ul:
        decoder1_o = decoder1_output[0, channel].flatten().detach().cpu().numpy()
        decoder2_o = decoder1_output[2, channel].flatten().detach().cpu().numpy()
    else:
        decoder1_o = decoder1_output[0, channel].flatten().detach().cpu().numpy()
        decoder2_o = decoder2_output[0, channel].flatten().detach().cpu().numpy()
    kde1 = gaussian_kde(decoder1_o)
    kde2 = gaussian_kde(decoder2_o)

    # Create a range of values where KDE is evaluated
    x_range = np.linspace(min(decoder1_o.min(), decoder2_o.min()), 
                        max(decoder1_o.max(), decoder2_o.max()), 1000)
    
    # Evaluate the density over this range
    kde1_density = kde1(x_range)
    kde2_density = kde2(x_range)

    # Plot the KDE of both decoders
    plt.figure(figsize=(8, 5))
    if ul:
        plt.plot(x_range, kde1_density, label='Labeled data KDE', color='blue')
        plt.plot(x_range, kde2_density, label='Unabeled data KDE', color='red')
    else:
        plt.plot(x_range, kde1_density, label='Decoder 1 KDE', color='blue')
        plt.plot(x_range, kde2_density, label='Decoder 2 KDE', color='red')
    plt.fill_between(x_range, kde1_density, alpha=0.3, color='blue')
    plt.fill_between(x_range, kde2_density, alpha=0.3, color='red')

    plt.title(f'KDE of Outputs in class {channel}')
    # plt.title(f'KDE of Outputs in class {channel} (iter {iter_num})')
    plt.xlabel('Output Value')
    plt.ylabel('Density')
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path, dpi=300)  # Save as PNG with 300 DPI
    plt.clf() 
    # plt.show()


def plot_DST(decoder1_output, decoder2_output, save_path, iter_num, channel = 0, ul = False):

    if ul:
        decoder1_o = decoder1_output[0, channel].flatten().detach().cpu().numpy()
        decoder2_o = decoder1_output[2, channel].flatten().detach().cpu().numpy()
    else:
        decoder1_o = decoder1_output[0, channel].flatten().detach().cpu().numpy()
        decoder2_o = decoder2_output[0, channel].flatten().detach().cpu().numpy()

    

    # kde1 = gaussian_kde(decoder1_o)
    # kde2 = gaussian_kde(decoder2_o)

    # Create a range of values where KDE is evaluated
    # x_range = np.linspace(min(decoder1_o.min(), decoder2_o.min()), 
    #                     max(decoder1_o.max(), decoder2_o.max()), 1000)
    
    # # Evaluate the density over this range
    # kde1_density = kde1(x_range)
    # kde2_density = kde2(x_range)

    # Plot the KDE of both decoders
    plt.figure(figsize=(8, 5))
    if ul:
        plt.hist(decoder1_o, bins=None, alpha=0.6, label='Labeled', color='blue')
        plt.hist(decoder2_o, bins=None, alpha=0.6, label='Unlabeled', color='red')
    else:
        plt.hist(decoder1_o, bins=None, alpha=0.6, label='Decoder 1', color='blue')
        plt.hist(decoder2_o, bins=None, alpha=0.6, label='Decoder 2', color='red')

    # if ul:
    #     plt.plot(x_range, kde1_density, label='Labeled data KDE', color='blue')
    #     plt.plot(x_range, kde2_density, label='Unabeled data KDE', color='red')
    # else:
    #     plt.plot(x_range, kde1_density, label='Decoder 1 KDE', color='blue')
    #     plt.plot(x_range, kde2_density, label='Decoder 2 KDE', color='red')
    # plt.fill_between(x_range, kde1_density, alpha=0.3, color='blue')
    # plt.fill_between(x_range, kde2_density, alpha=0.3, color='red')

    plt.title(f'Output distribution in class {channel}')
    # plt.title(f'KDE of Outputs in class {channel} (iter {iter_num})')
    plt.xlabel('Output Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path, dpi=300)  # Save as PNG with 300 DPI
    plt.clf() 
    plt.close()
    # plt.show()

