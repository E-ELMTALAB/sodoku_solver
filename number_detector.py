### number detection


import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)


# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

###############################################################
##### variables
###############################################################

red = (0 ,0 , 255)
IMAGE_PATHS = r'C:\Users\Morvarid\Pictures\Camera Roll\WIN_20230202_17_58_30_Pro.jpg'
PATH_TO_MODEL_DIR = r'C:\python\open_cv\opencv_practice\warp_bitwise_practice\my_model2\my_model'
PATH_TO_LABELS = r'C:\python\open_cv\opencv_practice\warp_bitwise_practice\my_model\label_map.pbtxt'
MIN_CONF_THRESH = float(0.60)

###############################################################
##### load model function (used for loading model in the beginning) 
###############################################################
def load_model():

    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

    print('Loading model...', end='')
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn

###############################################################
##### image classification  function 
###############################################################
def classify_digit(image_piece):

    predicted_num = 0 # this variable is use to store the predicted number for every image peice

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_piece)
    input_tensor = input_tensor[tf.newaxis, ...]

    ### detecting the give number image
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = { key: value[0, :num_detections].numpy() for key, value in detections.items() }
    predicted_num = int(detections['detection_classes'][0])

    return predicted_num


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

detect_fn = load_model() # load the model in this file instead of the main file

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

