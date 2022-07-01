# This is here to disable tensorflow's GPU warnings since I didn't have it configured
# on my laptop, and it was annoying to see them every time I launched the program
# Other than that this is not needed at all
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sudoku_solver import solve_sudoku
from sudoku_utils import locate_and_warp, draw_sudoku, sudoku_to_array, copy_board
from keras.models import load_model
import cv2
import sys

# Preprocess the arguments
image_path = sys.argv[1]
debug = True if len(sys.argv) == 3 and sys.argv[2] == "debug" else False

# Load the pretrained model and sudoku image
model = load_model("digit_classifier")
sudoku_image = cv2.imread(image_path)
if sudoku_image is None:
    raise Exception("Please put valid image")
image_size = (800, 800)
sudoku_image = cv2.resize(sudoku_image, image_size, interpolation=cv2.INTER_LINEAR)

# Create original and grayscale images with located and warped sudoku board
original_warped, grayscale_warped = locate_and_warp(sudoku_image, debug)

# Convert warped sudoku board to number array using the pretrained model and make a copy
cell_locations, board = sudoku_to_array(model, grayscale_warped)
board_copy = copy_board(board)

# If sudoku can be solved, show it with empty cells filled with solution,
# otherwise print that solution does not exist
if solve_sudoku(board, 0, 0):
    draw_sudoku(original_warped, cell_locations, board, board_copy)
else:
    print("Solution does not exist")
