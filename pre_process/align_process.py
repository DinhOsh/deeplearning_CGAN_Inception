import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import glob
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, default="align", choices=["align", "combine"])
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", help="path to folder containing combined/output image")
parser.add_argument("--size", type=int, default=256, help="set the image size to be aligned")

# parser.add_argument("--output_A_dir", help="path to folder A for training")
# parser.add_argument("--output_B_dir", help="path to folder B for training")
a = parser.parse_args()

orb = cv2.ORB_create(nfeatures=500)
LEN_OF_GOODS = 10


def calculate_location(img1, img2):

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    dx = 0.0
    dy = 0.0
    for m in matches[:LEN_OF_GOODS]:
        # print(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0], kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])
        dx += (kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0])/LEN_OF_GOODS
        dy += (kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])/LEN_OF_GOODS

    return int(dx), int(dy)


def align_and_resize(data_path, size):
    # resizing the image with the source image size

    folder_src = os.path.join(data_path, 'A')
    if not os.path.exists(folder_src):
        os.makedirs(folder_src)
    folder_tar = os.path.join(data_path, 'B')
    if not os.path.exists(folder_tar):
        os.makedirs(folder_tar)

    # Scan all files on target folder
    cnt = 1
    print("align and resizing for training...")
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    total = len(files)
    for f in files:

        fn, ext = os.path.splitext(f)
        if ext.lower() == '.jpg':
            png_fn = fn + '.png'
            png_path = os.path.join(data_path, png_fn)

            if os.path.isfile(png_path):
                print("%d / %d, File Name: %s\n" % (cnt, total, f))
                cnt += 1

                jpg_path = os.path.join(data_path, f)
                jpg_img = cv2.imread(jpg_path)
                png_img = cv2.imread(png_path, -1)

                # os.remove(jpg_path)
                # os.remove(png_path)

                if len(png_img.shape) != 3:  # Gray color image
                    print(" --- gray color --- ")
                    continue
                if png_img.shape[-1] == 3:  # Have no alpha channel
                    print(" --- There is no alpha channel --- ")
                    continue

                dx, dy = calculate_location(jpg_img, png_img)
                h, w = jpg_img.shape[:2]

                new_jpg_img = jpg_img

                trans_mat = np.float32([[1, 0, dx], [0, 1, dy]])
                new_png_img = cv2.warpAffine(png_img, trans_mat, (w, h))
                new_png_img = cv2.bitwise_and(new_jpg_img, new_jpg_img, None, new_png_img[:, :, 3])

                # Resize the image with (size , size)
                new_jpg_img = cv2.resize(new_jpg_img, (size, int(size * h / w)))
                new_png_img = cv2.resize(new_png_img, (size, int(size * h / w)))

                # aligned_jpg_canvas
                jpg_canvas = np.zeros((size, size, 3), dtype=np.uint8)
                # aligned png canvas
                png_canvas = np.zeros((size, size, 3), dtype=np.uint8)

                jpg_canvas[int(size / 2 - size * h / w / 2):int(size / 2 - size * h / w / 2) + int(size * h / w), :]\
                    = new_jpg_img
                png_canvas[int(size / 2 - size * h / w / 2):int(size / 2 - size * h / w / 2) + int(size * h / w), :]\
                    = new_png_img

                # save image in different folders
                new_jpg_path = os.path.join(folder_src, fn + '.jpg')
                cv2.imwrite(new_jpg_path, jpg_canvas)
                new_png_path = os.path.join(folder_tar, fn + '.jpg')
                cv2.imwrite(new_png_path, png_canvas)


def align_and_combine(data_path, output_path, size):
    # resizing the image with the source image size

    folder_out = output_path
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    # Scan all files on target folder
    cnt = 1
    print("align and resizing for training...")
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    total = len(files)
    for f in files:

        fn, ext = os.path.splitext(f)

        if ext.lower() == '.jpg':

            print("%d / %d, File Name: %s\n" % (cnt, total, f))
            cnt += 1

            jpg_path = os.path.join(data_path, f)
            jpg_img = cv2.imread(jpg_path)

            if len(jpg_img.shape) != 3:  # Gray color image
                print(" --- gray color --- ")
                jpg_img = cv2.cvtcolor(jpg_img, cv2.COLOR_GRAY2RGB)

            h, w = jpg_img.shape[:2]

            # Resize the image with (size , size)
            new_jpg_img = cv2.resize(jpg_img, (size, int(size * h / w)))

            # create a combined image with size (size, 2*size)
            combined = np.zeros((size, size * 2, 3), dtype=np.uint8)

            # aligned_jpg_canvas[
            combined[int(size / 2 - size * h / w / 2):int(size / 2 - size * h / w / 2) + int(size * h / w), :size] \
                = new_jpg_img
            combined[int(size / 2 - size * h / w / 2):int(size / 2 - size * h / w / 2) + int(size * h / w), size:] \
                = new_jpg_img

            # save combined image in different folders
            new_combined_path = os.path.join(folder_out, fn + '.png')
            cv2.imwrite(new_combined_path, combined)

if __name__ == '__main__':

    """
        python Pre_process/align_process.py \
          --mode align \
          --input_dir data/test/ \
          --output_dir data/test/output \
          --size 256
    """

    # opencv version is 3.2.0
    print(cv2.__version__)

    input_dir = a.input_dir
    if input_dir is None:
        raise Exception("input_dir not defined")
    output_dir = a.output_dir

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")
    # input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
    # if len(input_paths) == 0:
    #     raise Exception("input_dir contains no image files")

    if a.size is None:
        raise Exception("The alignment image size not defined.")

    # align for training
    if a.mode == "align":
        out_size = int(a.size)
        align_and_resize(input_dir, out_size)
        print("Successfully align the images for training")

    # align and combined for testing
    elif a.mode == "combine":
        out_size = int(a.size)
        align_and_combine(input_dir, output_dir, out_size)
        print("Successfully align the images for training")

