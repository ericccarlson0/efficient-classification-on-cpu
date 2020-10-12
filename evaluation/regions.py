# %% Imports; turn off warnings.

import os
import torch
import warnings

import numpy as np
import matplotlib.pyplot as plt

from models.util.images import fetch_images, divide_into_regions
from models.networks.shufflenet_custom import shufflenet_small

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")

# %% Set up directories.

BASE_DIR = os.sep + os.path.join("Users", "ericcarlson", "Desktop")
# GOOD_DIR = os.path.join(BASE_DIR, "Datasets", "RVL_CDIP")
GOOD_DIR = os.path.join(BASE_DIR, "Datasets", "ITD", "Good", "High_Density")
# BAD_DIR = os.path.join(BASE_DIR, "Datasets", "MSCOCO_2014")
BAD_DIR = os.path.join(BASE_DIR, "Datasets", "ITD", "Bad", "Web")
MODEL_NAME = "EMPTY1.pt"
STATE_DICT_PATH = os.path.join(BASE_DIR, "efficient-classification-on-cpu", "trained", MODEL_NAME)

print("Set up directories.")

# %% Create and load model.

model = shufflenet_small()
# TODO: Add back when appropriate model has been created...
# model.load_state_dict(torch.load(STATE_DICT_PATH))
model.eval()

print("Created and loaded model.")

# %% Fetch images and create regions.

relative_coord_list = [(.25, .75, .25, .75),
                       (.25, 1, .25, 1),
                       (0, .75, .25, 1),
                       (0, .75, 0, .75),
                       (.25, 1, 0, .75)]

num_images = 32

good_images = fetch_images(GOOD_DIR, extension="png", maxsize=num_images)
bad_images = fetch_images(BAD_DIR, extension="png", maxsize=num_images)
print("Fetched images.")

good_regions = []
for image in good_images:
    good_regions.extend(divide_into_regions(image, *relative_coord_list))

bad_regions = []
for image in bad_images:
    bad_regions.extend(divide_into_regions(image, *relative_coord_list))
print("Created regions.")

# %% Generate scores.

def score(module: torch.nn.Module, image: np.ndarray):
    mini_batch = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)
    output = module(mini_batch.float())
    probabilities = output.detach().numpy()

    # print(f"Probabilities: {probabilities}")
    return probabilities[0]

good_scores = [score(model, image) for image in good_images]
bad_scores = [score(model, image) for image in bad_images]
print("Classified *images*.")

good_reg_scores = [score(model, image) for image in good_regions]
bad_reg_scores = [score(model, image) for image in bad_regions]
print("Classified *image regions*.")

# %% Sanity check: assess misclassification by region.

# TODO: This assumes two classes...
def mc_by_region(image_scores: list, region_scores: list, n_regions: int):
    mc = [0] * n_regions

    for i, image_score in enumerate(image_scores):
        # image_score is a list of scores for each possible label.
        image_label = np.argmax(image_score)
        for j in range(n_regions):
            region_score = region_scores[i*n_regions + j]
            region_label = np.argmax(region_score)
            if image_label != region_label:
                mc[j] += 1

    return mc

misclassified = mc_by_region(good_scores, good_reg_scores, len(relative_coord_list))
print(f"""
    Misclassification by region ("Good" images): {misclassified}
""")

misclassified = mc_by_region(bad_scores, bad_reg_scores, len(relative_coord_list))
print(f"""
    Misclassification by region ("Bad" images): {misclassified}
""")

# %% Plot and show which regions contribute to classification.

edgecolor = [0, 0, 0]
color_matrix = [
    [0, 0, 1],
    [1, 0, 0]
]

for i in range(num_images):
    image = bad_images[i]
    h, w = image.shape[:2]
    extent = 0, h, 0, w
    fig = plt.figure(frameon=False)

    # base_image = plt.imshow(image[:, :, 0], cmap=plt.get_cmap("gray"))
    base_image = plt.imshow(image)
    plt.axis("off")

    for j, coords in enumerate(relative_coord_list):
        h1, h2 = int(h * coords[0]), int(h * coords[1])
        y_rect_coords = [h1, h2, h2, h1]
        w1, w2 = int(w * coords[2]), int(w * coords[3])
        x_rect_coords = [w1, w1, w2, w2]

        image_score = bad_scores[i]
        region_score = bad_reg_scores[i * len(relative_coord_list) + j]
        # color = np.matmul([region_score], color_matrix)
        bad_score, good_score = np.exp(region_score) / sum(np.exp(region_score))
        color = [bad_score, 0, good_score]
        print(color)

        plt.fill(x_rect_coords, y_rect_coords, alpha=0.25, color=color, edgecolor=edgecolor, lw=3)

plt.show()
