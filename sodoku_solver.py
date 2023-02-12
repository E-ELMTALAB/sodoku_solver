### sodoku solver
import box_finder_sliscer as box_finder
import numpy as np


# global board
# board = [
#     [7,8,0,4,0,0,1,2,0],
#     [6,0,0,0,7,5,0,0,9],
#     [0,0,0,6,0,1,0,7,8],
#     [0,0,7,0,4,0,2,6,0],
#     [0,0,1,0,5,0,9,3,0],
#     [9,0,4,0,6,0,0,0,5],
#     [0,7,0,3,0,0,0,1,2],
#     [1,2,0,0,0,7,4,0,0],
#     [0,4,9,2,0,6,0,0,7]
# ]

# def set_board(matrix):
#     board = matrix


def solve(bo): # this function has recursion, which its own function within itself. Every layer of function within each function, is an empty space after an empty space.
  find = find_empty(bo)
  if not find: return True # If it can't find an empty space, it stops the function. The 'solve' function will only return 'True' when the sudoku has been solved.
  else: row, col = find # retrieves the row and column of the empty space.
  
  for i in range (1,10): # 'i' (refered to as num in the valid function) refers to the numbers that are tested in the previously empty position. 
    if valid(bo, i, (row, col)):  # if the 'for loop' and 'valid' function find a valid entry, it
      bo[row][col] = i # ...will be inserted into the board, and...
    #   written_box = box_finder.write_on_box(image , row , col , i )
      
      if solve(bo): # ...it will restart the solve function on the recently updated board to find a valid value for the next empty space, but if the solve function has been declared 'True' because an empty space could not be found when find_empty was run at the beginning of the function, ...
        return True # ...this function will return 'True', causing the the function (itself) that called this line to continue running from the 'if statement' that called this line to return 'True', which causes this line to return 'True'. This will repeat until all of the previous recursive calls of this function have returned 'True'.
      bo[row][col] = 0 # Python will only run this code if it knows that the next/following recursive call has returned 'False'. If the code is running the last iteration of the 'for loop', the previous recursive call can return to the space that this recursive call is running, because this line declares this recursive call's space as zero (a.k.a., an empty value).  
  return False # The function will return 'False' if a valid value for a space could not be found (i.e., all the iterations for the space could not be found), causing it to continue running the 'for loop' of the previous space from the second 'if statement'.


def valid(bo, num, pos): #this checks if a value entry into a space is valid by checking if that same value is already in the same row, column, or box. It will return 'True' if it has found a valid solution.
  # Check row. # Because of how the solve function called pos, pos[0] refers to the row and pos[1] refers to the column.
  for i in range(len(bo[0])): # the 'i' refers to the x-value (or column) of the position.
    if bo[pos[0]][i] == num and pos[1] != i: # if each element in the row == recently added number as long as it isnt in the position that we just inserted.
      return False

  # Check column.
  for i in range(len(bo)):
    if bo[i][pos[1]] == num and pos[0] != i: 
      return False
  
  # Check Box of 3 by 3
  box_x = pos[1] // 3
  box_y = pos[0] // 3
  for i in range(box_y*3, box_y*3 + 3): # this loops through the 3 by 3 box to check whether we have the same element appearing twice.
    for j in range(box_x*3, box_x*3 + 3):
      if bo[i][j] == num and (i,j) != pos:
        return False
  return True # a function stops running when it reaches a return statement.


def print_board(bo, image , prev_board ):
#   for i in range(len(bo)): # for loop iterating through each row.  'i' is the row, and 'j' is each value in each row. 
#     if i % 3 == 0 and i != 0: print("- - - - - - - - - - - -")
    
#     for j in range(len(bo[0])): #iterating through each column of each row.
#       if j % 3 == 0 and j != 0: print(" | ", end="")
        
#       if j == 8: print(bo[i][j])
#       else: print(bo[i][j], end=" ")

    for row in range(9):
        for column in range(9):
            if  not prev_board[row][column]:
              written_box = box_finder.write_on_box(image , row , column , bo[row][column])




    bo = np.array(bo , np.int8)
    print(bo)
    return image


def find_empty(bo): #this finds empty spaces.
  for i in range(len(bo)):
    for j in range(len(bo[0])):
      if bo[i][j] == 0:
        return (i, j) # row, then column of empty spaces.
  return None


# execution
# print_board(board)
# print('\n\t {}\n'.format(solve(board)))
# print_board(board)