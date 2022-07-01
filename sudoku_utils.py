from imutils.perspective import four_point_transform
from keras_preprocessing.image import img_to_array
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


# Locates and warps sudoku from given image using four point perspective transform
# to obtain top-down bird's eye view of the sudoku
def locate_and_warp(image, debug=False):
    # Convert the image to grayscale and blur it slightly
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (7, 7), 3)
    # Apply adaptive thresholding and then invert the threshold map
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = cv2.bitwise_not(threshold)

    if debug:
        cv2.imshow('Threshold image', threshold)
        cv2.waitKey(0)

    # Find contours in the threshold image and sort them by size in
    # descending order
    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Initialize a contour that corresponds to the puzzle outline
    sudoku_contours = None
    # Loop over the contours
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        # If our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            sudoku_contours = approx
            break

    # If the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if sudoku_contours is None:
        raise Exception("Could not find Sudoku puzzle outline.")

    # Apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the sudoku
    warped_rgb = four_point_transform(image, sudoku_contours.reshape(4, 2))
    warped_gs = four_point_transform(grayscale, sudoku_contours.reshape(4, 2))

    if debug:
        cv2.imshow('Warped original image', warped_rgb)
        cv2.waitKey(0)

    return warped_rgb, warped_gs


# Extracts digit from given cell
def extract_digit(cell):
    # Apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    threshold = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    threshold = clear_border(threshold)

    # Find contours in the threshold cell
    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # If no contours were found than this is an empty cell
    if len(contours) == 0:
        return None
    # Otherwise, find the largest contour in the cell and create a
    # mask for the contour
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(threshold.shape, dtype="uint8")
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = threshold.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # If less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None
    # Apply the mask to the threshold cell
    digit = cv2.bitwise_and(threshold, threshold, mask=mask)
    return digit


# Converts warped grayscale sudoku image to digit array using pretrained model
def sudoku_to_array(model, warped_gs):
    # Initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int")
    # Sudoku is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped_gs.shape[1] // 9
    stepY = warped_gs.shape[0] // 9
    # Initialize a list to store the (x, y)-coordinates of each cell location
    cellLocations = []

    # Loop over the grid locations
    for y in range(0, 9):
        # Initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # Compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            # Add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            # Crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped_gs[startY:endY, startX:endX]
            digit = extract_digit(cell)
            # Verify that the digit is not empty
            if digit is not None:
                # Resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                # Classify the digit and update the Sudoku board with the
                # prediction
                prediction = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = prediction
        # Add the row to our cell locations
        cellLocations.append(row)

    board = list(board)
    return cellLocations, board


# Copies the array and returns a copy (.copy() or [:]) didn't work since
# it also copied the references and solve() function changed both arrays
def copy_board(board):
    board_copy = []
    for i in range(9):
        row = []
        for j in range(9):
            row.append(board[i][j])
        board_copy.append(row)
    return board_copy


# Draws the solved sudoku onto the warped original image
def draw_sudoku(warped_rgb, cellLocations, board, board_copy):
    # Loop over the cell locations and board
    curr_row, curr_col = -1, -1
    for (cellRow, boardRow) in zip(cellLocations, board):
        curr_row += 1
        # Loop over each individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            curr_col += 1
            if board_copy[curr_row][curr_col % 9] != 0:
                continue

            # Unpack the cell coordinates
            startX, startY, endX, endY = box
            # Compute the coordinates of where the digit will be drawn
            # on the output sudoku image
            textX = int((endX - startX) * 0.4)
            textY = int((endY - startY) * -0.15)
            textX += startX
            textY += endY
            # Draw the result digit on the Sudoku puzzle image
            cv2.putText(warped_rgb, str(digit), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the output image
    cv2.imshow("Result", warped_rgb)
    cv2.waitKey(0)
