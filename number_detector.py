### number detection

"""
Object Detection (On Image) From TF2 Saved Model
=====================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
import cv2
# from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
# IMAGE_PATHS = r'C:\python\open_cv\object_detection\065cfcf34ad578c1704fb47cbe34a047.jpg'


# # PROVIDE PATH TO MODEL DIRECTORY
# PATH_TO_MODEL_DIR = r'C:\python\trained_models\my_model'

# # PROVIDE PATH TO LABEL MAP
# PATH_TO_LABELS = r'C:\python\trained_models\my_model\label_map.pbtxt'

IMAGE_PATHS = r'C:\Users\Morvarid\Pictures\Camera Roll\WIN_20230202_17_58_30_Pro.jpg'


# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = r'C:\python\open_cv\opencv_practice\warp_bitwise_practice\my_model2\my_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = r'C:\python\open_cv\opencv_practice\warp_bitwise_practice\my_model\label_map.pbtxt'


# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

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

    # LOAD LABEL MAP DATA FOR PLOTTING

detect_fn = load_model()

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
red = (0 ,0 , 255)

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




# print('Running inference for {}... '.format(IMAGE_PATHS), end='')
# cap = cv2.VideoCapture(0)
# width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

def classify_digit(image_piece):
    # print("trying to classify")

    predicted_num = 0
    # while cv2.waitKey(1) != ord('q'):

        #   _ , image = cap.read()
        # image = cv2.imread(IMAGE_PATHS)
    image = image_piece
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    # detect_fn = load_model()
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    #   num_detections = int(detections.pop('num_detections'))
    #   detections = {key: value[0, :num_detections].numpy()
    #                 for key, value in detections.items()}
    #   detections['num_detections'] = num_detections
    #   center = ( int(detections['detection_boxes'][0][3] * height) , int(detections['detection_boxes'][0][0] * width))
    #   boxes = detections['detection_boxes']
    #   classes = detections['detection_classes']  
    #   scores = detections['detection_scores']

    # print()

    # if center :

    ### positions of the detection
    #   positions = boxes[0]
    #   # position = detections['detection_boxes'][detections['detection_scores'] > 0.8]
    #   (xmin, xmax, ymin, ymax) = (int(positions[1]*width), int(positions[3]*width), int(positions[0]*height) , int(positions[2]*height))
    #   (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    # print(detections['detection_classes'][detections['detection_scores'] > 0.8])
    predicted_num = int(detections['detection_classes'][0])
    # print(detections['detection_classes'][0][0])
    # print(classes)
    # cv2.circle(image , (left ,top ) , 50 , red , -1 )

    #   for i,b in enumerate(boxes):
    #     # if classes[0][i] == 2: # ch
    #     if (scores[i] >= 0.5) :
    #       mid_x = (boxes[i][1]+boxes[i][3])/2
    #       mid_y = (boxes[i][0]+boxes[i][2])/2
    #       # array_ch.append([mid_x, mid_y])
    #       cv2.circle(image,(int(mid_x*width),int(mid_y*height)), 30, (0,0,255), -1)

    # detection_classes should be ints.
    # detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #         image_with_detections,
    #         detections['detection_boxes'],
    #         detections['detection_classes'],
    #         detections['detection_scores'],
    #         category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=4,
    #         min_score_thresh=0.3,
    #         agnostic_mode=False)

    # print('Done')
    # DISPLAYS OUTPUT IMAGE
    # cv2.imshow("the image" , image_with_detections)
  # cv2.imshow(" camera " , image)
    return predicted_num
# CLOSES WINDOW ONCE KEY IS PRESSED
