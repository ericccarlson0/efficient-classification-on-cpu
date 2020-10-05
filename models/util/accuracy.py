import numpy as np
import torch
import time

from torch.utils.data import DataLoader
from models.util.dataset import StandardDataset
from models.util.visualization import show_images

def generate_misclassified(model: torch.nn.Module, image_ids, label_mappings, dataset_dir: str, num_classes: int = 2):

    dataset = StandardDataset(image_ids, label_mappings, dataset_dir=dataset_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    prediction_list = []
    target_list = []

    with torch.no_grad():
        for curr_image, curr_label in loader:
            output = model(curr_image.float())
            prediction = np.argmax(output.numpy(), axis=1)

            prediction_list.append(prediction[0])
            target_list.append(curr_label.item())

    misclassified = []
    fn_list = [0] * num_classes

    i = 0
    for prediction, target in zip(prediction_list, target_list):
        if prediction != target:
            misclassified.append(image_ids[i])
            fn_list[target] += 1
        i += 1

    return misclassified, fn_list

def visualize_misclassified(**kwargs):
    start_time = time.time()
    misclassified, fn_list = generate_misclassified(**kwargs)
    duration = time.time() - start_time
    print(f"It took {duration: .3f} seconds to classify.")

    show_images(misclassified, torch_data_dir=kwargs["dataset_dir"])

    return misclassified, fn_list
