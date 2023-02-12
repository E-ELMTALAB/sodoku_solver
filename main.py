### main ###

import box_finder_sliscer as box_finder
import sodoku_solver as sod_solver
import number_detector as detector
import cv2
import numpy as np

################################################################
##### variables
################################################################

n = predicted_number = 0
new_found = 1
frames_passed = flag = 0
warped_copy = rewarped = None
array = np.zeros((9,9) , np.int8)
prev_board = np.zeros((9,9) , np.int8)

###############################################################
##### main function 
###############################################################

if __name__ == '__main__':


    cap = cv2.VideoCapture(0)  # setting to read the camera
    while cv2.waitKey(1) != ord('q'): # loop until the key "q" has been pressed
        
    #### first operations on the image 
        _ , image = cap.read()
        image_copy = image.copy()
        image_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray ,(5 , 5) ,1 )
        image_threshold = cv2.adaptiveThreshold(image_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19, 8)
        image_canny = cv2.Canny(image_threshold , 10 , 50)
        image_dilate = cv2.dilate(image_canny , (5 , 5) , iterations=4)

        ### find the four ponits of the square
        square_found , four_points , image = box_finder.find_four_points(image , image_dilate , False , False)
        

        ###############################################################
        ##### if square was found  
        ###############################################################
        if square_found:

            frames_passed = frames_passed + 1      # used to track the frames passed to classify the numbers in the board
            warped , points1 , points2 = box_finder.warp(image , four_points)   # warpping the image

            ### slice the board into 9*9 peices
            boxes = box_finder.slice(warped)
            boxes = np.array(boxes)

            ################## classifying cells after 30 frames ########################

            if frames_passed == 30 :    # after 30 frames passed (used for detecting the image clearly)

                if new_found:  # if you found a new board 

                    flag = 1
                    warped_copy = warped.copy()

                    ################### classifying each box of the board #####################
                    for i in range(9):
                        for j in range(9):

                            ### checking if the box is empty 
                            box = boxes[i][j][25:75 , 25:75]
                            _ , thresh_box = cv2.threshold(box , 100 , 255 , cv2.THRESH_BINARY_INV)
                            none_zero_counter = np.count_nonzero(thresh_box)

                            ### if there was a number in the box classify it
                            if none_zero_counter > 300 :
                                predicted_number = (detector.classify_digit(boxes[i][j]))

                            ### if not put zero in that position
                            else: 
                                predicted_number = 0

                            ### filling the arrays
                            array[i][j] = predicted_number
                            if flag:
                                prev_board[i][j] = predicted_number
                                
                    new_found = flag = 0
                frames_passed = 0

                ### solve the sodoku
                sod_solver.solve(array)

            ### if the array was not empty write it on the image
            if np.sum(array) > 0 :
                sod_solver.print_board(array , warped , prev_board)

            rewarped = box_finder.rewarp(image , points1 , points2 , warped)    # rewarp the image to put it on the original image
            image = rewarped

        ### if no new square was found 
        else:       
            new_found = 1
            flag = 0
            array = np.zeros_like(array)


    #### showing the processed image 
        cv2.imshow("image" , image)