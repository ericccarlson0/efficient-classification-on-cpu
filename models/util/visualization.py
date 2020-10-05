import matplotlib.pyplot as plt
import os
import torch

def retrieve_torch_img(img_id: str, torch_data_dir: str):
    img_path = os.path.join(torch_data_dir, img_id + ".pt")
    return torch.load(img_path)

def retrieve_numpy_img(img_id: str, torch_data_dir: str):
    torch_img = retrieve_torch_img(img_id, torch_data_dir)
    return torch_img.numpy().transpose((1, 2, 0))

def show_images(image_ids, torch_data_dir: str, start_dex: int = 0, ndisplay: int = 36, ncols: int = 6, cmap=None):
    nrows = int(ndisplay / ncols + 0.5)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    for row in range(nrows):
        for col in range(ncols):
            ax[row, col].axis("off")

            offset = row*ncols + col
            if offset >= ndisplay:
                continue

            show_id = image_ids[start_dex + offset]
            show_img = retrieve_numpy_img(show_id, torch_data_dir=torch_data_dir)

            ax[row, col].imshow(show_img, cmap=cmap)

    plt.show()

def check_classification(model: torch.nn.Module, image_id: str, torch_data_dir: str):
    torch_img = retrieve_torch_img(image_id, torch_data_dir=torch_data_dir)
    numpy_img = torch_img.numpy().transpose((1, 2, 0))

    with torch.no_grad():
        mini_batch = torch_img.unsqueeze(0)
        output = model(mini_batch.float())

        probabilities = output.numpy()
        class_probabilities = [f"{probability: .4f}" for probability in probabilities[0]]
        # TODO: This needs to change if there are more than just a handful of classes.
        text = " ".join(class_probabilities)

    plt.imshow(numpy_img)
    plt.axis("off")
    plt.xlabel(text)
    plt.show()
