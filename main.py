### main ###
import box_finder_sliscer as box_finder
import sodoku_solver as sod_solver
import number_detector as detector
import cv2
import numpy as np

n = predicted_number = 0
new_found = 1
frames_passed = flag = 0
warped_copy = rewarped = None

array = np.zeros((9,9) , np.int8)
prev_board = np.zeros((9,9) , np.int8)
board = [
    [7,8,0,4,0,0,1,2,0],
    [6,0,0,0,7,5,0,0,9],
    [0,0,0,6,0,1,0,7,8],
    [0,0,7,0,4,0,2,6,0],
    [0,0,1,0,5,0,9,3,0],
    [9,0,4,0,6,0,0,0,5],
    [0,7,0,3,0,0,0,1,2],
    [1,2,0,0,0,7,4,0,0],
    [0,4,9,2,0,6,0,0,7]
]

# board = [
#     [0,0,0,9,0,2,0,0,0],
#     [0,4,0,0,0,0,0,5,0],
#     [0,0,2,0,0,0,3,0,0],
#     [2,0,0,0,0,0,0,0,7],
#     [0,0,0,4,5,6,0,0,0],
#     [6,0,0,0,0,0,0,0,9],
#     [0,0,7,0,0,0,8,0,0],
#     [0,3,0,0,0,0,0,4,0],
#     [0,0,2,0,0,0,7,0,0]
# ]

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    while cv2.waitKey(1) != ord('q'):
        
    #### first operations on the image 
        _ , image = cap.read()
        image_copy = image.copy()
        image_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray ,(5 , 5) ,1 )
        image_threshold = cv2.adaptiveThreshold(image_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19, 8)
        image_canny = cv2.Canny(image_threshold , 10 , 50)
        image_dilate = cv2.dilate(image_canny , (5 , 5) , iterations=4)

        square_found , four_points , image = box_finder.find_four_points(image , image_dilate , False , False)
        
        if square_found:
            # new_found = 1
            frames_passed = frames_passed + 1
            warped , points1 , points2 = box_finder.warp(image , four_points)

            boxes = box_finder.slice(warped)
            boxes = np.array(boxes)
            sod_solver.solve(board)
            # sod_solver.print_board(board , warped)
            # num_0_5 = (detector.classify_digit(boxes[0][5]))
            # num_0_8 = (detector.classify_digit(boxes[0][8]))

            # written_box = box_finder.write_on_box(warped , 0 , 8 , 69 )
            # written_box = box_finder.write_on_box(warped , 0 , 7 , num_0_8 )
            # print("predicted : " + str(predicted_number))

            ### wrting the detected number on every cell
            if frames_passed == 30 :
                if new_found:
                    flag = 1
                    warped_copy = warped.copy()
                    for i in range(9):
                        for j in range(9):
                            # print("new_found : " + str(new_found))
                            # print("i : " + str(i) + " j : " + str(j))

                            box = boxes[i][j][25:75 , 25:75]
                            _ , thresh_box = cv2.threshold(box , 100 , 255 , cv2.THRESH_BINARY_INV)
                            none_zero_counter = np.count_nonzero(thresh_box)
                            if none_zero_counter > 300 :

                                predicted_number = (detector.classify_digit(boxes[i][j]))
                                # written_box = box_finder.write_on_box(warped_copy , i , j , predicted_number )

                            else: 
                                # written_box = box_finder.write_on_box(warped_copy , i , j , 0 )
                                predicted_number = 0

                            array[i][j] = predicted_number
                            if flag:
                                prev_board[i][j] = predicted_number
                                
                            # print(predicted_number)
                    new_found = 0
                    flag = 0
                    # print("now new_found : " + str(new_found))
                frames_passed = 0
                # image = warped_copy

                ### solve the sodoku
                sod_solver.solve(array)
                print(array)
                print("\n")
                print("previous board :")
                print(prev_board)

            # _ , box_thresh = cv2.threshold(boxes[0][8] , 100 , 255 , cv2.THRESH_BINARY_INV)
            # cv2.imshow("box_thresh" , box_thresh)
            # print(np.count_nonzero(box_thresh))

            ### if the array was not empty
            if np.sum(array) > 0 :
                sod_solver.print_board(array , warped , prev_board)
            # boxes2 = np.reshape(boxes , (9 , 9))
            # written_box = box_finder.write_on_box(warped , 4 , 8 , 6)
            # written_box = box_finder.write_on_box(warped , 8 , 0 , 3)
            # written_box = box_finder.write_on_box(warped , 0 , 8 , 7)
            # written_box = box_finder.write_on_box(warped , 8 , 8 , 2)

            rewarped = box_finder.rewarp(image , points1 , points2 , warped)




            # if flag :
                # cv2.imshow("warped" , warped_copy)
            new_box = boxes[0][6][20:80 , 20:80]
            _ , thresh_box = cv2.threshold(new_box , 100 , 255 , cv2.THRESH_BINARY_INV)
            none_zero_counter = np.count_nonzero(thresh_box)
            # print(none_zero_counter)
            # cv2.imshow("new_box" , new_box)
            # cv2.imshow("thersh_new_box" , thresh_box)
            # cv2.imshow("slice[0]" , boxes[2][7])
            # print(boxes[0][2].shape)
            # cv2.imshow("written_box" , written_box)
            # cv2.imshow("rewarped" , rewarped)
            image = rewarped
            # print(boxes)
        else:
            new_found = 1
            flag = 0
            array = np.zeros_like(array)


    #### showing the processed image 
        cv2.imshow("image" , image)
        # print(array)