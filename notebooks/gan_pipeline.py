# %% Import packages.

import os
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.networks.dcgan_custom import Generator, Discriminator
from models.util.initialization import gan_init

print("Imported packages.")

# %% Set up directories.

print("Setting up directories...")

BASE_DIR = os.sep + os.path.join("Users", "ericcarlson", "Desktop")
CDIP_DIR = os.path.join(BASE_DIR, "Datasets", "RVL_CDIP", "images")
COCO_DIR = os.path.join(BASE_DIR, "Datasets", "MSCOCO_2014")

# %% Set up training parameters.


print("Setting up training parameters...")

num_epochs = 1024
max_batches = 512

num_workers = 4
batch_size = 128
image_size = 64

nc = 3
nz = 128
ngf = 64
ndf = 64

lr = 2e-4
beta1 = .5
beta2 = .999

# %% Set up Datasets and DataLoaders.

print("Setting up Datasets and DataLoaders...")

preprocess = transforms.Compose([
    # TODO: Edit this in the future. The center just seems more reliable for now.
    transforms.CenterCrop(image_size * 2),
    transforms.RandomCrop(image_size),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(CDIP_DIR, preprocess)
cdip_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

coco_dataset = datasets.ImageFolder(COCO_DIR, preprocess)
coco_dataloader = DataLoader(coco_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# %% Show some original data.

batch_ex = next(iter(cdip_dataloader))
grid = np.transpose(torchvision.utils.make_grid(
    batch_ex[0], padding=True, normalize=True
), (1, 2, 0))

plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(grid)
plt.show()

# %% Create Generator and Discriminator.

print("Creating Generator and Discriminator...")

gNet = Generator(nz=nz, ngf=ngf, nc=nc)
gNet.apply(gan_init)

dNet = Discriminator(nc=nc, ndf=ndf)
dNet.apply(gan_init)

# %% Set up criterion and optimizers.

print("Setting up loss criterion and optimizer...")

criterion = nn.BCELoss()
gOptimizer = optim.Adam(gNet.parameters(), lr=lr, betas=(beta1, beta2))
dOptimizer = optim.Adam(dNet.parameters(), lr=lr, betas=(beta1, beta2))

base_noise = torch.randn(4, nz, 1, 1)

# %% TRAINING

# TODO: change.
log_freq = 24
iterations = 0

good_label = .9
bad_label = .1

g_losses = []
d_losses = []
count = 0

for epoch in range(num_epochs):
    good_iterator = iter(cdip_dataloader)
    bad_iterator = iter(coco_dataloader)
    while True:
        good_data = next(good_iterator)[0]
        bad_data = next(bad_iterator)[0]
        data = torch.cat((good_data, bad_data))

        # max_batches is required here because we use batch_size to decide the size of "labels"
        # otherwise, we would use data.size(0)
        if count >= max_batches:
            break
        count += 1

        # Discriminator: real batch.
        dNet.zero_grad()
        outputs = dNet(data).view(-1)

        good_labels = torch.full((batch_size,), good_label, dtype=torch.float)
        bad_labels = torch.full((batch_size,), bad_label, dtype=torch.float)
        labels = torch.cat((good_labels, bad_labels))

        d_loss = criterion(outputs, labels)
        d_loss.backward()

        D_x = outputs.mean().item()

        # Discriminator: generated batch.
        noise = torch.randn(batch_size, nz, 1, 1)
        labels = torch.full((batch_size,), bad_label, dtype=torch.float)

        g_outputs = gNet(noise)
        outputs = dNet(g_outputs.detach()).view(-1)

        d_g_loss = criterion(outputs, labels)
        d_g_loss.backward()

        D_G_z1 = outputs.mean().item()
        d_loss = d_loss + d_g_loss

        dOptimizer.step()

        # Generator
        gNet.zero_grad()
        labels.fill_(good_label)

        outputs = dNet(g_outputs).view(-1)

        g_loss = criterion(outputs, labels)
        g_loss.backward()

        D_G_z2 = outputs.mean().item()

        gOptimizer.step()

        # Don't start at zero for logging.
        if count % log_freq == 0:
            print(f"epoch {epoch}, iter {count}")
            print(f"D loss: {d_loss: .3f}, G loss: {g_loss: .3f}")
            print(f"D(x): {D_x: .3f}, D(G(z)): {D_G_z1: .3f}, {D_G_z2: .3f}")

            generated = gNet(base_noise).detach()
            fake_grid = torchvision.utils.make_grid(generated)
            plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
            plt.title(f"{epoch}.{count}")
            plt.axis('off')
            plt.show()

        g_losses.append(g_loss)
        d_losses.append(d_loss)

        iterations += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Training")
plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.ylabel("LOSS")
plt.legend()
plt.show()

# %% Show some generated images.

with torch.no_grad():
    fake_batch = gNet(torch.randn(16, nz, 1, 1))
    fake_grid = torchvision.utils.make_grid(fake_batch, nrow=4, padding=0, normalize=False)
    plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
    plt.show()