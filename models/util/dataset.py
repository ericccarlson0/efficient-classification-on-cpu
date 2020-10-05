import os
import torch
from torch.utils.data import Dataset

# This Dataset loads saved torch Tensors. These Tensors must all be saved in the same directory in the format
# <dataset_dir>/<image_id>.pt (where there is a dict, labels, from the image_id to the correct classification).
class StandardDataset(Dataset):
    def __init__(self, ids, labels, dataset_dir: str, tsfm=None):
        self.ids = ids
        self.labels = labels
        self.dataset_dir = dataset_dir
        self.tsfm = tsfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, dex):
        if torch.is_tensor(dex):
            dex = dex.tolist()
        image_id = self.ids[dex]

        filename = os.path.join(self.dataset_dir, image_id + ".pt")
        X = torch.load(filename)

        if self.tsfm:
            X = self.tsfm(X)

        y = self.labels[image_id]

        return X, y
