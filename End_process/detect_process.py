import argparse
import glob
import os
from os import listdir
from os.path import isfile, join
import sys
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import math

parser = argparse.ArgumentParser()
parser.add_argument("--origin_dir", help="path to folder containing original images")
parser.add_argument("--output_dir", help="path to folder containing ML processed images")
parser.add_argument("--size", default=256, help="size of ML processed image")
# parser.add_argument("--mode", default="scan", help="process the folder or indivisul image file")

a = parser.parse_args()

KERNEL = np.ones((5, 5), np.uint8)


def rect_points(ori_im, rm_thumb_im):

    # convert to gray, removed background image (size: 256 * 256)
    rm_thumb_gray = cv2.cvtColor(rm_thumb_im, cv2.COLOR_RGB2GRAY)
    # threshold the gray image with gaussian filtering
    # rm_bin = cv2.adaptiveThreshold(rm_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret, rm_thumb_bin = cv2.threshold(rm_thumb_gray, 10, 255, cv2.THRESH_BINARY)
    # soil erosion
    rm_thumb_bin = cv2.erode(rm_thumb_bin, KERNEL, iterations=1)
    # dilation, opposite of erosion
    rm_thumb_bin = cv2.dilate(rm_thumb_bin, KERNEL, iterations=1)

    # resizing to fit the original size
    h, w, chs = ori_im.shape

    th_h, th_w = rm_thumb_bin.shape[:2]

    M = np.float32([[float(w) / th_w, 0, 0], [0, float(w) / th_w, int((h - float(th_h * w) / th_w) / 2)]])
    mask = cv2.warpAffine(rm_thumb_bin, M, (w, h))

    # get contour from mask image
    im, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # merging contours
    rect = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        rect.append([x, y, x+w, y+h])
    rect = np.array(rect)
    pick = non_max_suppression(rect, probs=None, overlapThresh=0.65)

    max_square = 0.0
    res_rect = []
    for (x1, y1, x2, y2) in pick:
        square = math.fabs((x1 - x2) * (y1 - y2))



        if max_square < square:
            max_square = square
            res_rect = [(x1, y1), (x2, y2)]

    # cv2.rectangle(ori_im, res_rect[0], res_rect[1], (0, 0, 255))

    # removed background image with the same size of original image
    rm_bk_im = np.zeros(ori_im.shape)
    for ch in range(chs):
        rm_bk_im[:, :, ch] = ori_im[:, :, ch] * (1.0 - (mask[:, :] / 255.0))

    return res_rect, rm_bk_im


def color_detect(rm_thumb):

    col = (0, 0, 0)
    return col


def text_detect(ori):

    text = ""
    return text


def end_process(ori_im, rm_thumb_im):

    res_dict = {}
    rect, rm_im = rect_points(ori_im, rm_thumb_im)
    color = color_detect(rm_thumb_im)
    text = text_detect(ori_im)

    res_dict["rect_points"] = rect
    res_dict["color"] = color
    res_dict["text"] = text

    return res_dict


def scan_folder(ori_dir, out_dir):

    cnt = 1
    # Scan all files on origin folder
    files = [f for f in listdir(ori_dir) if isfile(join(ori_dir, f))]
    total = len(files)
    for f in files:

        fn, ext = os.path.splitext(f)

        if ext.lower() == '.jpg':

            rm_fn = fn + '-outputs.png'
            rm_path = os.path.join(out_dir, rm_fn)  # file path of removed backgroud and cropped

            if os.path.isfile(rm_path):
                sys.stdout.write("%d / %d, File Name: %s\n" % (cnt, total, f))
                cnt += 1

                ori_path = os.path.join(ori_dir, f)
                ori_img = cv2.imread(ori_path)
                rm_thumb_img = cv2.imread(rm_path, -1)
                result_dict = end_process(ori_img, rm_thumb_img)

                print(result_dict["rect_points"])
                print(result_dict["color"])


if __name__ == '__main__':

    """
        python End_process/detect_process.py --origin_dir --output_dir
    """

    # opencv version is 3.2.0
    print("--- Settings ---")
    print("opencv version : ", cv2.__version__)

    origin_dir = a.origin_dir
    output_dir = a.output_dir
    # get all images from origin_dir and output_dir
    origin_paths = glob.glob(os.path.join(a.origin_dir, "*.jpg"))
    output_paths = glob.glob(os.path.join(a.output_dir, "*.png"))
    print("Input directories ")
    print("    Origin Directories :", a.origin_dir, len(origin_paths), " files")
    print("    Output Directories :", a.output_dir, len(output_paths), " files")

    output_size = a.size
    print("size of ML processed :", output_size)

    if origin_dir is None:
        raise Exception("input_dir not defined")
    if output_dir is None:
        raise Exception("input_dir not defined")

    if len(origin_paths) == 0:
        raise Exception("There is no image file to process")
    if len(output_paths) == 0:
        raise Exception("There is no image file which ML processed")

    # end_process()
    print("End Processing...")
    scan_folder(origin_dir, output_dir)
    print("Successfully finished the scaning the folder")
