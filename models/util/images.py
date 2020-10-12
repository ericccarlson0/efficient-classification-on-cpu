import glob
import numpy as np
import sys
import torch

from skimage import io, transform

# Returns a list of all images in a directory with the provided extension.
def fetch_images(directory: str, extension: str = "jpg", maxsize: int = sys.maxsize):
    images = []
    count = 0
    for filename in glob.iglob(directory + f"**/*.{extension}", recursive=True):
        if count >= maxsize:
            break
        images.append(to_three_channel(io.imread(filename)))

    return images

# Returns np.ndarray or torch.Tensor with three channels.
# (from one that could have had one or four channels, or even just two dimensions)
def to_three_channel(image):
    if isinstance(image, np.ndarray):
        return to_three_channel_numpy(image)
    elif isinstance(image, torch.Tensor):
        return to_three_channel_torch(image)

def to_three_channel_torch(tensor: torch.Tensor):
    if tensor.ndimension() == 2:
        w, h = tensor.shape
        return torch.zeros((3, w, h)) + tensor.unsqueeze(0)
    else:
        w, h = tensor.shape[1:]
        if tensor.shape[0] == 1:
            tensor = torch.zeros(3, w, h) + tensor
        elif tensor.shape[0] == 4:
            tensor = tensor[:3, :, :]
        return tensor

def to_three_channel_numpy(ndarray: np.ndarray):
    w, h = ndarray.shape[:2]
    if ndarray.ndim == 2:
        ndarray = np.zeros((w, h, 3)) + np.expand_dims(ndarray, axis=2)
    elif ndarray.shape[2] == 1:
        ndarray = np.zeros((w, h, 3)) + ndarray
    elif ndarray.shape[2] == 4:
        ndarray = ndarray[:, :, :3]
    return ndarray

# TODO: Do we want this to work for Tensors, too?
# Returns a single-channel numpy array according to specified color conversion.
def to_single_channel(image, color: str):
    assert image.shape[2] >= 3

    if color == 'R':
        image = image[:, :, 0]
    elif color == 'G':
        image = image[:, :, 1]
    elif color == 'B':
        image = image[:, :, 2]
    else:
        image = np.dot(image, [0.2989, 0.5870, 0.1140])

    image = np.expand_dims(image, axis=2)

    return image

# TODO: It's probably a good idea to just have divide_into_four and similar methods operate on just individual images...
# Returns a list of quadrants using a list of original images.
def divide_into_four(full_images):
    new_images = []
    for img in full_images:
        half_w = int(img.shape[0]/2)
        half_h = int(img.shape[1]/2)
        new_images.append(img[half_w:, :half_h, :])
        new_images.append(img[:half_w, :half_h, :])
        new_images.append(img[:half_w, half_h:, :])
        new_images.append(img[half_w:, half_h:, :])
    return new_images

# Returns a list with corner quadrants and a center quadrant using a list of original images.
def divide_into_five(full_images):
    new_images = []
    for img in full_images:
        half_w = int(img.shape[0]/2)
        half_h = int(img.shape[1]/2)
        quarter_w = int(half_w/2)
        quarter_h = int(half_h/2)
        new_images.append(img[quarter_w:half_w + quarter_w, quarter_h:half_h + quarter_h])
        new_images.append(img[half_w:, :half_h, :])
        new_images.append(img[:half_w, :half_h, :])
        new_images.append(img[:half_w, half_h:, :])
        new_images.append(img[half_w:, half_h:, :])
    return new_images

# Returns a list with images cropped according to a list of relative coordinates, where (0, 0) would represent
# the top-left corner, (1, 1) would represent the bottom-right corner, and all coordinates are between 0 and 1.
def divide_into_regions(image, *relative_coord_list):
    regions = []
    w, h = image.shape[:2]
    for relative_coord in relative_coord_list:
        w1 = int(w * relative_coord[0])
        w2 = int(w * relative_coord[1])
        h1 = int(h * relative_coord[2])
        h2 = int(h * relative_coord[3])
        regions.append(image[w1:w2, h1:h2])
    return regions

# Returns a randomly cropped numpy ndarray. Its dimensions are side_len x side_len.
def random_crop(image: np.ndarray, side_len: int):
    old_w, old_h = image.shape[:2]

    if old_w <= side_len or old_h <= side_len:
        raise ValueError(f"The entered size {side_len} should be less than both width, {old_w} and height, {old_h}")

    left = np.random.randint(0, old_w - side_len)
    top = np.random.randint(0, old_h - side_len)

    image = image[left:left + side_len, top:top + side_len]

    return image

# Returns a numpy ndarray cropped in the center. Its dimensions are side_len x side_len.
def center_crop(image: np.ndarray, side_len: int):
    old_w, old_h = image.shape[:2]

    if old_w <= side_len or old_h <= side_len:
        raise ValueError(f"The entered size {side_len} should be less than both width, {old_w} and height, {old_h}")

    left = int((old_w - side_len) / 2)
    top = int((old_h - side_len) / 2)

    image = image[left:left + side_len, top:top + side_len]

    return image

# Re-sizes a numpy ndarray by maintaining the aspect ratio and scaling the shortest side to side_len.
def resize(image: np.ndarray, side_len: int):
    old_w, old_h = image.shape[:2]
    if old_w > old_h:
        new_w, new_h = int(side_len * old_w / old_h), side_len
    else:
        new_w, new_h = side_len, int(side_len * old_h / old_w)

    return transform.resize(image, (new_w, new_h))

# Distorts a numpy ndarray RGB channels.
def distort_rgb(image: np.ndarray):
    mask = np.random.random((3, 3))
    return np.dot(image, mask)

# Returns the sizes, widths, and heights of a list of images.
def get_sizes(image_list: list):
    widths = []
    heights = []
    sizes = []
    for image in image_list:
        width, height = image.shape[:2]
        widths.append(width)
        heights.append(height)
        sizes.append(width * height)
    return sizes, widths, heights
