# outline

Based on [Image to Image Translate(CGAN)](https://bitbucket.org/Din_Osh/outline.git) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)

Tensorflow implementation of pix2pix.  Learns a mapping from input images to output images, like these examples from the original paper:

The processing speed on a GPU with cuDNN was equivalent to the Torch implementation in testing.

## Setup

### Prerequisites
- Tensorflow 1.0.0
- opencv 3.2.0

### Recommended
- Linux with Tensorflow GPU edition + cuDNN
- CUDA 8.0
- cuDNN-8.0 5.1

### Project Details

- CGAN_model
    tools
        process.py
        split.py
    pix2pix.py

- Inception_model

- pre_processing
    align_process.py
- data
    sample-cars/
        front_3_quarter
### Train the model

# CGAN_model

```sh
# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
python CGAN_model/pix2pix.py \
  --mode train \
  --output_dir CGAN_model/model/front_3_quarter \
  --max_epochs 500 \
  --input_dir data/sample-cars/front_3_quarter/combined/train \
  --which_direction AtoB \
  --checkpoint CGAN_model/model/front_3_quarter

python CGAN_model/pix2pix.py \
  --mode train \
  --output_dir CGAN_model/model/front_3_quarter \
  --max_epochs 500 \
  --input_dir data/sample-cars/front_3_quarter/combined/train \
  --which_direction AtoB \
  --checkpoint CGAN_model/model/front_3_quarter

# test the model
python CGAN_model/pix2pix.py \
  --mode test \
  --output_dir ...\
  --input_dir  ...\
  --checkpoint ...
```


## Datasets and Trained Models

The data format used by this program is the same as the original image to image translate format, which consists of images of input and desired output side by side like:

Some datasets have been made available by the authors of the pix2pix paper.  To download those datasets, use the included script `tools/download-dataset.py`.  There are also links to pre-trained models alongside each dataset, note that these pre-trained models require the Tensorflow 0.12.1 version of pix2pix.py since they have not been regenerated with the 1.0.0 release:

| dataset | example |
| --- | --- |


```sh
# align and resize the source images
python Pre_process/align_process.py \
  --input_dir data/sample-cars/front_3_quarter \
  --mode align \
  --size 256

# Combine the A folder images and B folder images
python CGAN_model/tools/process.py \
  --input_dir data/sample-cars/front_3_quarter/A \
  --b_dir data/sample-cars/front_3_quarter/B \
  --operation combine \
  --output_dir data/sample-cars/front_3_quarter/combined

# Split into train/val set
python CGAN_model/tools/split.py \
  --dir data/sample-cars/front_3_quarter/combined
```

The folder `./combined` will now have `train` and `val` subfolders that you can use for training and testing.


#### Creating image pairs from existing images

If you have two directories `a` and `b`, with corresponding images (same name, same dimensions, different data) you can combine them with `process.py`:

```sh
python tools/process.py \
  --input_dir a \
  --b_dir b \
  --operation combine \
  --output_dir c
```

This puts the images in a side-by-side combined image that `pix2pix.py` expects.

## Training

### Image Pairs
For normal training with image pairs, you need to specify which directory contains the training images, and which direction to train on.  The direction options are `AtoB` or `BtoA`
Train the model (this took about 1 ~ 2 days depending on GPU, on CPU you will be waiting for a bit)

Training from init status
```sh
python CGAN_model/pix2pix.py \
  --mode train \
  --output_dir CGAN_model/model/front_3_quarter \
  --max_epochs 500 \
  --input_dir data/sample-cars/front_3_quarter/combined/train \
  --which_direction AtoB \
```
Training go on from the checkpoint
```sh
python CGAN_model/pix2pix.py \
  --mode train \
  --output_dir CGAN_model/model/front_3_quarter \
  --max_epochs 500 \
  --input_dir data/sample-cars/front_3_quarter/combined/train \
  --which_direction AtoB \
  --checkpoint CGAN_model/model/front_3_quarter
```

In this mode, image A is the black and white image (lightness only), and image B contains the color channels of that image (no lightness information).

## Testing

Testing is done with `--mode test`.  You should specify the checkpoint to use with `--checkpoint`, this should point to the `output_dir` that you created previously with `--mode train`:

Test the model with checkpoints
```sh
python CGAN_model/pix2pix.py \
  --mode test \
  --output_dir input_dir data/sample-cars/front_3_quarter/combined/test_result\
  --input_dir input_dir data/sample-cars/front_3_quarter/combined/val\
  --checkpoint CGAN_model/model/front_3_quarter
```

The testing mode will load some of the configuration options from the checkpoint provided so you do not need to specify `which_direction` for instance.

## Code Validation

Validation of the code was performed on a Linux machine with a Ubuntu 14.04, Nvidia GTX 980 Ti GPU and an AWS P2 instance with a K80 GPU.

```sh
# move to the project directory
cd outline

# preprocessing
python Pre_process/align_process.py \
  --mode combine \
  --input_dir data/temp/ \
  --output_dir data/temp/output \
  --size 256

# main process/ML processing
python CGAN_model/pix2pix.py \
  --mode test \
  --output_dir data/temp/output\
  --input_dir data/temp/output\
  --checkpoint CGAN_model/model/front_3_quarter

# endprocesing
python End_process/detect_process.py \
  --origin_dir data/temp/ \
  --output_dir data/temp/output/images \
  --size 256
```


python Pre_process/align_process.py --input_dir data/test/ --mode align --size 256

## Citation
If you use this code for your research, please cite the paper this code is based on: <a href="https://arxiv.org/pdf/1611.07004v1.pdf">Image-to-Image Translation Using Conditional Adversarial Networks</a>:

```
@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}
```

## Acknowledgments
This is a port of [pix2pix](https://github.com/phillipi/pix2pix) from Torch to Tensorflow.  It also contains colorspace conversion code ported from Torch.  Thanks to the Tensorflow team for making such a quality library!  And special thanks to Phillip Isola for answering my questions about the pix2pix code.
