# %% Import packages.
import configparser
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as tf

from models.util.visualization import show_images, check_classification
from models.networks.shufflenet_custom import shufflenet_small
from models.util.dataset import StandardDataset
from models.util.accuracy import generate_misclassified

from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings("ignore")

print("Imported packages.")

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

print("Set up directories.")

# %% Retrieve image IDs and tensors.

image_ids = []
label_mappings = {}
csv_dataset = pd.read_csv(CSV_DIR)

for i in range(len(csv_dataset)):
    image_id = csv_dataset.iloc[i, 0]
    label = int(csv_dataset.iloc[i, 1])

    image_ids.append(image_id)
    label_mappings[image_id] = label

print("Retrieved image IDs and tensors.")

# %% Divide image IDs into Train, Val, Test sets.

train_ids, val_ids = train_test_split(image_ids, test_size=.20)
val_ids, test_ids = train_test_split(val_ids, test_size=.50)

print(f"Train, Val, Test: {len(train_ids)}, {len(val_ids)}, {len(test_ids)}")

# %% Check original images.

show_image_num = 36
start_dex = np.random.randint(0, len(train_ids) - show_image_num)
show_images(train_ids, torch_data_dir=TENSORS_DIR, start_dex=start_dex, ndisplay=show_image_num)

print("Checked original images.")

# %% Set up training parameters.

lr = 1e-4
# lr_decay = 0.95
dropout_prob = 0.50
prune_prob = 0.10
prune_mod = sys.maxsize

batch_size = 32
num_workers = 2
num_classes = 2
num_epochs = 16
finetune_depth = 6

print("Set up training parameters.")

# %% Define training.

def train_epoch(max_batches: int = sys.maxsize):
    losses = 0
    count = 0

    for loaded_tensor, loaded_label in loader:
        if count >= max_batches:
            break

        optimizer.zero_grad()
        output = model(loaded_tensor)
        loss = criterion(output, loaded_label.long())
        losses += loss.item()

        loss.backward()
        optimizer.step()

        count += 1
        if (count % 32) == 0:
            print(f"{count} batches...")

    print(f"Aggregate loss over {count} batches: {losses: .3f}")

print("Defined training.")

# %% Define pruning.

def prune_model():
    if prune_mod < num_epochs and (epoch % prune_mod) == 0:
        # TODO: Fill with real code.
        pass

print("Defined pruning.")

# %% Determine how to LOAD, TRAIN, and/or RECORD results.

# This would be a string to represent a directory to load parameters.
load_model_dir = None
pretrain = True
train_model = True
save_model = False
add_hparams = False

# This is used for recording results and saving the model.
model_type = "ShuffleNet_Custom_DocumentClass"

writer = SummaryWriter(TB_LOGDIR)
hparams = {"MODEL_TYPE": model_type, "LEARNING_RATE": lr, "DROPOUT_PROB": dropout_prob}

# %% Create model. Optionally load model.

model = shufflenet_small()

if load_model_dir:
    model.load_state_dict(torch.load(load_model_dir))

print("Created model.")

# %% Set up loss criterion, optimizer, LR scheduler.

criterion = nn.CrossEntropyLoss()
# Can add betas and/or AMSGrad later...
optimizer = optim.Adam(params=model.parameters(), lr=lr)
# lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

print("Set up loss criterion, optimizer, lr scheduler.")

# %% Set up preprocessing, Dataset, DataLoader

# This is the normalization used for ImageNet. It may not be appropriate depending on the use-case.
normalize = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Preprocessing does not include cropping or resizing. The choice is made, here, that this should be done before saving
# the tensors, in order to train as efficiently as possible.
preprocess = tf.Compose([
    normalize
])

torch_dataset = StandardDataset(train_ids, label_mappings, tsfm=preprocess, dataset_dir=TENSORS_DIR)
loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

print("Set up preprocessing, Dataset, DataLoader.")

# %% TRAIN MODEL.

print("\nSTARTING EPOCHS")

for epoch in range(train_model * num_epochs):
    print(f"\nEPOCH {epoch}:")

    train_epoch()
    # prune_model()
    # lr_scheduler.step(epoch=epoch)

    _, fn_list = generate_misclassified(model=model, image_ids=val_ids, label_mappings=label_mappings,
                                        dataset_dir=TENSORS_DIR, num_classes=num_classes)
    val_accuracy = sum(fn_list) / len(val_ids)
    print(f"Validation accuracy: {val_accuracy: .4f}")

    for j in range(8):
        curr_id = val_ids[j]
        curr_label = label_mappings[curr_id]
        check_classification(model, image_id=curr_id, torch_data_dir=TENSORS_DIR)

print("Model has been created.")

# %% Save model to appropriate directory (optional).

# TODO: There could be pruning or something like that before we save the model.
# That would need to be dealt with...
if save_model:
    state_dict_path = os.path.join(SAVED_MODEL_DIR,
                                   f"{model_type}_time={float(time.time()): .2f}.pt")

    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved model to {state_dict_path}.")

# %% Record results as hparams (optional).

# TODO: Generate accuracy on Val and Test sets when training is over.
val_acc = 0
test_acc = 0
total_acc = 0

if add_hparams:
    metrics = {"hparam/val_accuracy": val_acc, "hparam/test_accuracy": test_acc, "hparam/total_accuracy": total_acc}
    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

writer.close()

# %% Trace the model (optional).

trace_model = False

# TODO: Finish this. And traced model on a fixed input is almost certainly not enough.
if trace_model:
    ex_input = torch.rand((2, 3, 244, 244))
    traced_model = torch.jit.trace(model, ex_input)
