### box finder and sliscer

import cv2
import numpy as np 

### variables
red = (0 ,0 ,255 )
blue = (255 ,0 ,0)
green = (0 ,255 ,0 )
magenta = (255 ,0 ,255)
warped_h = 900
warped_w = 900
global detected 

### the video caputre 
cap = cv2.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))

###############################################################
##### writing the specific number on each cell 
###############################################################

def write_on_box(image , row , column , text):

    const = 100
    y = (row *const) + 70   # set the y position 
    x = (column *const) + 30    # set the x position

    cv2.putText(image , str(text) , (x , y) , cv2.FONT_HERSHEY_SIMPLEX , 2 , magenta , 3) 
    x = y = 0

    return image

###############################################################
##### slicing the board to 9*9 cells 
###############################################################

def slice(image):

    rows = np.array_split(image , 9)    # split the image vertically
    w, h = 9, 9
    boxes = [[0 for x in range(w)] for y in range(h)] 

    for i in range(9):

        columns = np.array_split(rows[i] , 9 , axis=1)  # split the image horizantaly
        for j in range(9):

            boxes[i][j] = columns[j]    # put each cell in boxes list

    return boxes 

###############################################################
##### warp the image to classify and write on cells 
###############################################################

def warp(image , four_points):

        # four_points = rearange_points(points)
        points1 = np.float32(four_points)
        points2 = np.float32([[0, 0], [warped_w, 0], [warped_w, warped_h],[0, warped_h]])

        # warping the image and putting text on it
        transformation_matrix = cv2.getPerspectiveTransform(points1, points2)
        warped_img = cv2.warpPerspective(image, transformation_matrix, (warped_w, warped_h))

        return warped_img , points1 , points2

###############################################################
##### rewarping the image to put on the main image
###############################################################

def rewarp(image , points1 , points2 , warped_img):

        #creating the mask 
        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners = np.int32(points1)
        mask = cv2.fillConvexPoly(mask, roi_corners, (255, 255, 255))
        mask = cv2.bitwise_not(mask)

        #bitwise_and operation
        bitwise_and_img = cv2.bitwise_and(mask , image)

        # rewarped doc image
        transformation_matrix = cv2.getPerspectiveTransform(points2, points1)
        rewarped_img = cv2.warpPerspective(warped_img, transformation_matrix, (width, height))

        # bitwise or to get the final image 
        final_img = cv2.bitwise_or(rewarped_img , bitwise_and_img)
        image = final_img
        
        return image
    
###############################################################
##### used for arraging the points 
##### ( used for stabling the four points )
###############################################################

def rearange_points(points):
    
    sorted_pts = np.zeros((4, 2), dtype="int32")
    s = np.sum(points, axis=1)
    sorted_pts[0] = points[np.argmin(s)]
    sorted_pts[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    sorted_pts[1] = points[np.argmin(diff)]
    sorted_pts[3] = points[np.argmax(diff)]

    return sorted_pts

###############################################################
##### function for finding the four points of square 
###############################################################

def find_four_points(image , processed_image , draw= False , circle = False):

    #### needed variable
    square_points = [] # where we keep our four points of square
    final_img = None

    #### finding the contours 
    contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # cv2.drawContours(image, cnt, -1, (0, 255, 255), 3)

    #### searching for the biggest rectangle or the square in the image
    epsilon = 0.1 * cv2.arcLength(cnt , True)
    approximations = cv2.approxPolyDP(cnt, epsilon, True)
    i, j = approximations[0][0] 
    if len(approximations) == 4:

        ######## square detected
        detected = True

        ######## setting th opsitoins
        left_top = (int(approximations[0][0][0]) , int(approximations[0][0][1]))
        left_bottom = (int(approximations[1][0][0]) , int(approximations[1][0][1]))
        right_bottom = (int(approximations[2][0][0]) , int(approximations[2][0][1]))
        right_top = (int(approximations[3][0][0]) , int(approximations[3][0][1]))
        square_points = [left_top , right_top , right_bottom , left_bottom]
        square_points = rearange_points(square_points)

        ######## if drawing the four points was intended
        if draw and not circle:

            cv2.line(image , square_points[0] , left_bottom , red , 3) # left_top
            cv2.line(image ,square_points[1] , right_bottom , blue , 3) # right_top
            cv2.line(image , square_points[2] , right_top , green , 3) # right_ bottom
            cv2.line(image , square_points[3] , left_top , magenta , 3) # left_bottom

        if draw and circle :

            cv2.circle(image , square_points[0] , 10 , red , -1)
            cv2.circle(image , square_points[1] , 10 , blue , -1)
            cv2.circle(image , square_points[2] , 10 , green , -1)
            cv2.circle(image , square_points[3] , 10 , magenta , -1)

    else:
        detected = False

    return detected , square_points , image


###############################################################
##### main function 
###############################################################

if __name__ == '__main__':

    while cv2.waitKey(1) != ord('q'):
        
    #### first operations on the image 
        _ , image = cap.read()
        image_copy = image.copy()
        image_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray ,(5 , 5) ,1 )
        image_threshold = cv2.adaptiveThreshold(image_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19, 8)
        image_canny = cv2.Canny(image_threshold , 10 , 50)
        image_dilate = cv2.dilate(image_canny , (5 , 5) , iterations=4)

        four_points , image = find_four_points(image_copy , image_dilate , True , True)


    #### showing the processed image 
        cv2.imshow("image" , image)
