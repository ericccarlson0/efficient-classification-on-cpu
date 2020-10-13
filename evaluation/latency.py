# %% Imports; turn off warnings.

import matplotlib.pyplot as plt
import numpy as np

import os
import time
import torch
import warnings

from models.util.images import fetch_images, divide_into_four, get_sizes
from models.networks.shufflenet_custom import shufflenet_small

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")

# %% Set up directories.

BASE_DIR = os.sep + os.path.join("Users", "ericcarlson", "Desktop")
HR_DIR = os.path.join(BASE_DIR, "Datasets", "HighRes")
LR_DIR = os.path.join(BASE_DIR, "Datasets", "MSCOCO_2014")
MODEL_NAME = "EMPTY1.pt"
STATE_DICT_PATH = os.path.join(BASE_DIR, "efficient-classification-on-cpu", "trained", MODEL_NAME)

print("Set up directories.")

# %% Create and load model.

model = shufflenet_small()
# TODO: Add back when appropriate model has been created...
# model.load_state_dict(torch.load(STATE_DICT_PATH))
model.eval()

print("Created and loaded model.")

# %% Define functions to evaluate latency.

num_images = 8

def test_latency(module: torch.nn.Module, tensor: torch.Tensor):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    start_time = time.time()
    _ = module(tensor.float())
    return time.time() - start_time

def plot(sizes: list, latencies: list):
    plt.ylabel("LATENCY (secs)")
    plt.xlabel("PIXELS")

    for size, latency in zip(sizes, latencies):
        plt.plot(size, latency, "ko")
    plt.show()

print("Defined functions to evaluate latency.")

# %% Evaluate High-Resolution latencies. Plot.

hr_images = fetch_images(HR_DIR, extension="jpg", maxsize=num_images)
hr_latencies = [
    test_latency(model, torch.from_numpy(image.transpose((2, 0, 1))))
    for image in hr_images
]
print("Checked latencies for High-Resolution images.")

hr_quads = divide_into_four(hr_images)
hr_quad_latencies = [
    test_latency(model, torch.from_numpy(image.transpose((2, 0, 1))))
    for image in hr_quads
]
print("Checked latencies for quadrants of High-Resolution images.")

hr_sizes, hr_widths, hr_heights = get_sizes(hr_images)
plot(hr_sizes, hr_latencies)

hr_quad_sizes, _, _ = get_sizes(hr_quads)
plot(hr_quad_sizes, hr_quad_latencies)
print("Plotted data for High-Resolution images.")

# %% Evaluate Low-Resolution latencies. Plot.

lr_images = fetch_images(LR_DIR, extension="jpg", maxsize=num_images)
lr_latencies = [
    test_latency(model, torch.from_numpy(image.transpose((2, 0, 1))))
    for image in lr_images
]
print("Checked latencies for Low-Resolution images.")

lr_quads = divide_into_four(lr_images)
lr_quad_latencies = [
    test_latency(model, torch.from_numpy(image.transpose((2, 0, 1))))
    for image in lr_quads
]
print("Checked latencies for quadrants of Low-Resolution images.")

lr_sizes, lr_widths, lr_heights = get_sizes(lr_images)
plot(lr_sizes, lr_latencies)

lr_quad_sizes, _, _ = get_sizes(lr_quads)
plot(lr_quad_sizes, lr_quad_latencies)
print("Plotted data for Low-Resolution images.")

# %% Show some statistics of the images that were checked.

# Helper method to reduce duplication.
def print_stats(stat_list: list, category_tag: str, stat_tag: str):
    stat_mean, stat_std = np.average(stat_list), np.std(stat_list)
    print(f"{category_tag} {stat_tag}: AVG of {stat_mean: .3f}, STD of {stat_std: .3f}")

# Helper method to reduce duplication.
def print_image_stats(latencies, widths, heights, category_tag: str):
    print_stats(latencies, stat_tag="LATENCY", category_tag=category_tag)
    print_stats(widths, stat_tag="WIDTH", category_tag=category_tag)
    print_stats(heights, stat_tag="HEIGHT", category_tag=category_tag)

print_image_stats(hr_latencies, hr_widths, hr_heights, category_tag="HIGH RES")

print_image_stats(lr_latencies, lr_widths, lr_heights, category_tag="LOW RES")
