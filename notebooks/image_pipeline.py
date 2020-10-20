# %% Import packages.

import os
import sys
import time
import torch
import torchvision
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models.networks.shufflenet_custom as shuffle
import models.networks.mobilenet_custom as mobile
from models.util.visualization import show_tensor
from models.util.dataset import StandardDataset
from models.util.train import train

from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")

print("Imported packages...")

# %% Set up directories.

# The directory that is beneath all the rest. Optional.
BASE_DIR = os.sep + os.path.join("Users", "ericcarlson", "Desktop")

# Where this project is stored.
LOCAL_DIR = os.path.join(BASE_DIR, "efficient-classification-on-cpu")

# Where all the images are stored as torch Tensors.
TENSORS_DIR = os.path.join(BASE_DIR, "Datasets", "torch_data")

# The csv that contains mappings from ID to label.
CSV_DIR = os.path.join(BASE_DIR, "Datasets", "csv", "ITD.csv")

# Where to save data associated with training.
TB_LOGDIR = os.path.join(LOCAL_DIR, "tensorboard")

# Where to save trained models.
SAVED_MODEL_DIR = os.path.join(LOCAL_DIR, "trained")

# TODO: Turn this into an actual, comprehensive error.
if not BASE_DIR:
    raise NotADirectoryError("Directories need to be set up.")

print("Set up directories...")

# %% Retrieve image IDs and tensors.

image_ids = []
label_mappings = {}
csv_dataset = pd.read_csv(CSV_DIR)

for i in range(len(csv_dataset)):
    image_id = csv_dataset.iloc[i, 0]
    label = int(csv_dataset.iloc[i, 1])

    image_ids.append(image_id)
    label_mappings[image_id] = label

print("Retrieved IDs and labels...")

# %% Divide image IDs into Train, Val, Test sets.

train_ids, val_ids = train_test_split(image_ids, test_size=.10)
val_ids, test_ids = train_test_split(val_ids, test_size=.50)

print("Divided the data...")
print(f"Data: {len(train_ids)}, {len(val_ids)}, {len(test_ids)}")

# %% Set up training parameters.

lr = 1e-4
lr_decay = 0.95
dropout_prob = 0.50
prune_prob = 0.10
prune_mod = sys.maxsize

batch_size = 16
num_workers = 2
num_classes = 2
num_epochs = 16
finetune_depth = 2

print("Set up training parameters...")

# %% Determine how to LOAD, TRAIN, and RECORD results.

# This would be the directory of a state dictionary.
state_dict_dir = None
pretrain = True
train_model = True
save_model = False

model_type = "MobileNet"
task_name = "Document_Classification"
# This is used for recording results and saving the model.
model_name = f"{model_type}_{task_name}"

writer = SummaryWriter(TB_LOGDIR)
hparams = {"MODEL_NAME": model_name, "LEARNING_RATE": lr, "DROPOUT_PROB": dropout_prob}

# %% Create model. Optionally load model.

if model_type == "MobileNet":
    MOBILENET_DIR = os.path.join(LOCAL_DIR, "models", "networks", "mobilenetv3_small.pth.tar")
    model = mobile.mobilenet_small(pretrained=True, net_dir=MOBILENET_DIR)
    mobile.prepare_for_finetune(model, depth=finetune_depth)

# Default to ShuffleNet.
else:
    model = shuffle.shufflenet_small()
    shuffle.prepare_for_finetune(model, depth=finetune_depth)

if state_dict_dir:
    model.load_state_dict(torch.load(state_dict_dir))

print("Created model...")

# %% Set up loss criterion, optimizer, LR scheduler.

criterion = nn.CrossEntropyLoss()
# Can add betas and/or AMSGrad later...
optimizer = optim.Adam(params=model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

print("Set up loss criterion, optimizer, LR scheduler...")

# %% Set up preprocessing, Dataset, DataLoader

# This is the normalization used for ImageNet. It may not be appropriate depending on the use-case.
normalize = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Preprocessing does not include cropping or resizing. The choice is made, here, that this should be done before saving
# the tensors, in order to train as efficiently as possible.
preprocess = None

datasets = {
    'train': StandardDataset(train_ids, label_mappings, tsfm=preprocess, dataset_dir=TENSORS_DIR),
    'val': StandardDataset(val_ids, label_mappings, tsfm=preprocess, dataset_dir=TENSORS_DIR)
}
dataloaders = {
    x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for x in ['train', 'val']
}
dataset_sizes = {
    x: len(datasets[x])
    for x in ['train', 'val']
}

print("Set up preprocessing, Dataset, DataLoader...")

# %% Check images from DataLoader.

inputs, labels = next(iter(dataloaders['train']))
grid = torchvision.utils.make_grid(inputs,
                                   nrow=int(np.sqrt(batch_size)))

fig, ax = plt.subplots(1, figsize=(10, 10))
show_tensor(grid, ax=ax)
plt.show()

print("Checked loaded images...")

# %% TRAIN model.

print("Starting training...")

if train_model:
    train(model=model,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=lr_scheduler,
          loaders=dataloaders,
          sizes=dataset_sizes,
          writer=writer,
          num_epochs=num_epochs)

print("Model has been created!")

# %% Save model to appropriate directory (optional).

# TODO: There could be pruning or something before we save the model, which would need to be dealt with.
if save_model:
    state_dict_path = os.path.join(SAVED_MODEL_DIR,
                                   f"{model_name}_t{float(time.time()): .3f}.pt")

    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved model to {state_dict_path}.")

# %% Record results as hparams.

# TODO: Generate accuracy on Val and Test sets when training is over.
val_acc = 0
test_acc = 0
total_acc = 0


metrics = {"hparam/val_accuracy": val_acc, "hparam/test_accuracy": test_acc, "hparam/total_accuracy": total_acc}
writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

writer.close()

# %% Trace the model (optional).

trace_model = False

# TODO: Finish this. And traced model on a fixed input is almost certainly not enough.
if trace_model:
    ex_input = torch.rand((2, 3, 244, 244))
    traced_model = torch.jit.trace(model, ex_input)
