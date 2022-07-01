# Sudoku Picture Solver
## Table of contents
* [General info](#general-info)
* [Illustration](#illustration)
* [Setup](#setup)
* [Example Usage](#example-usage)
* [Project status](#project-status)

## General info
Program takes a photo of a Sudoku puzzle as an input and returns it with empty cells filled with solution. It uses OpenCV to locate the puzzle and warps it with the four point transform to obtain bird's eye view of the puzzle (sudoku_utils.py). Then it extracts the digits from each individual cells using trained convolutional neural  network on MNIST digit dataset (CNN_Digits.py) and creates array representing the sudoku. At last it solves the sudoku array using some logic with recursion (sudoku_solver.py) and draws the digits back into the image (sudoku_utils.py).

## Illustration
![Sample input and it's corresponding output](https://i.imgur.com/QHBbkJ0.jpg)

## Setup
1. Clone the repository and unpack it
2. Install requirements with "pip install -r requirements.txt"
3. Run main.py with python (more details at [Example Usage](#example-usage))

## Example Usage
*python main.py "Sudoku Images/sudoku1.jpg" debug*

Program has one mandatory parameter (path to image) and one optional (debug)

Path to image should contain path to a valid sudoku image, otherwise program ends with exception

Debug parameter is used to obtain intermediate steps from the image processing (thresholded image and warped image after four point transform) to see if something went wrong

## Project status
I consider this project done since it's purpose was to try and create a project with multiple parts put together, which it satisfied. What could be improved upon later is better handling of arguments and non-sudoku images.



