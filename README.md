# Efficient Classification on CPU
This project is not necessarily one that attempts to optimize machine learning pipelines for use on 
CPU, but rather one whose purpose is to train and evaluate efficient neural networks without the use 
of a GPU. It started as a project whose purpose was to compare efficient neural networks (such as 
*Shufflenet*, *MobileNet*, and *EfficientNet*) for use on various image classification tasks. It does 
not go beyond standard image classification as of now.

The following sections are dedicated to explaining how to use the pipeline for yourself. I do not 
expect many besides myself to actually use this pipeline, but I hope this serves to explain some 
idiosyncrasies.

## How to use the Pipeline.
The training pipeline is located at `notebooks/image_pipeline`. There are two main steps that must 
be taken before this can be used, however. Step one is to add your local directories to the section 
titled "Set up directories". These point to your data, csv, and logging for TensorBoard. Step two is 
to generate your csv and to preprocess, augment, and save your data as `torch.Tensor` objects using 
`torch.save`.

## How to generate data for use in this Pipeline.
The main idiosyncratic element of this project has to do with data loading. The pipeline works by 
loading a set of IDs and the labels associated with those IDs. These IDs are then used to retrieve 
`torch.Tensor` objects that have been saved, all in the same directory, as `<image_id>.pt`.

Data is prepared by generating IDs for images, saving those images as `<image_id>.pt`, and adding to 
a csv that associates IDs with labels. One should preprocess and augment the images before saving 
them (explained in the next paragraph). The reason for structuring the data in such a way is that we 
need to use a form of lazy loading, unless we want to exhaust all of our memory on a list of images. 
One way to do this is for the `torch.utils.data.Dataset` implementation to load images directly from 
the filesystem (as opposed to loading them from a `list` that has been prepared programmatically). 
There probably better ways to do this, perhaps using `generators`... alas, I am but a naive college 
student.

Finally, preprocessing and augmentation should be done outside of pipeline because the alternatives 
force one to perform these CPU-intensive processes within the training process itself (either within 
the `Dataset` implementation or within `train_epoch`). This slows down training substantially.
Thanks for reading!
