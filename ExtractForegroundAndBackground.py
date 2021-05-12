import wget
from tensorflow.keras.datasets import cifar10
from pixellib.semantic import semantic_segmentation
import pixellib
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import glob


# Global Variables
base_path = os.getcwd()
pascal_segmentation_model = base_path + \
    "/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"

isDownloaded = False
if not os.path.isfile(pascal_segmentation_model):
    try:
        url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5'
        wget.download(url, pascal_segmentation_model)
        isDownloaded = True
    except Exception as e:
        print('Segmentation Model Not available. Aborting...')
        raise e

# load pascal segmentation model
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model(pascal_segmentation_model)


cifarLabels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

cifarImageCounts = {key: 1 for key in range(10)}


def ExtractForegroundAndBackground(img, label, imgCount, output_path):
    filename = f"{output_path}/{label}_{imgCount}_original.jpg"

    # saving originam image
    plt.imsave(filename, img)

    # extract foreground and background
    segment_image.segmentAsPascalvoc(
        filename, output_image_name="mask.png")

    newmask = cv.imread('mask.png', 0)

    mask = np.zeros(img.shape[:2], np.uint8)

    # saving foreground
    mask[newmask == 0] = 0
    mask[newmask != 0] = 1
    foreground = img*mask[:, :, np.newaxis]
    filename = f"{output_path}/{label}_{imgCount}_foreground.jpg"
    plt.imsave(filename, foreground)

    # saving background
    mask[newmask == 0] = 1
    mask[newmask != 0] = 0
    background = img*mask[:, :, np.newaxis]
    filename = f"{output_path}/{label}_{imgCount}_background.jpg"
    plt.imsave(filename, background)


def ProcessCifarData():

    global cifarImageCounts

    # load cifar data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    train_path = base_path + "/data/train"
    # Ensure that output path exist
    os.makedirs(train_path, exist_ok=True)

    train_images = 4  # x_train.shape[0]

    # generating training data
    for i in range(train_images):
        label = y_train[i][0]
        ExtractForegroundAndBackground(x_train[i], {cifarLabels[label]}, {
                                       cifarImageCounts[label]}, train_path)
        cifarImageCounts[label] += 1

    # Uncomment to process test images
    # test_path = base_path + "/data/test"
    # os.makedirs(test_path, exist_ok=True)
    # cifarImageCounts = {key: 1 for key in range(10)}
    # test_images = x_test.shape[0]
    # for i in range(test_images):
    #     label = y_test[i][0]
    #     ExtractForegroundAndBackground(x_test[i], label, test_path)
    #     cifarImageCounts[label] += 1


def ProcessLocalImages():

    os.chdir(base_path + "/images")
    train_path = base_path + "/data/local"
    # Ensure that output path exist
    os.makedirs(train_path, exist_ok=True)

    for file in glob.glob("*.png"):
        img = Image.open(file)
        ExtractForegroundAndBackground(np.asarray(img), file, 1, train_path)


# Processing Cifar dataset
ProcessCifarData()

# Processing local images: uncomment below code to run for images in local folder
#ProcessLocalImages()