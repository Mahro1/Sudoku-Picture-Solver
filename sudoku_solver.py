# Utility function to print out the sudoku grid
def print_board(grid):
    for row in grid:
        print(row)


# Function to check if number can be put on given row, column and 3x3 grid
def check_constraints(grid, row, col, num):
    # Check row
    for x in range(9):
        if grid[row][x] == num:
            return False
    # Check column
    for x in range(9):
        if grid[x][col] == num:
            return False

    # Check the 3x3 grid
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True


# Function that recursively solves the sudoku board using backtracking
def solve_sudoku(grid, row, col):
    # If we got to the end of the board, the sudoku is solved
    if row == 8 and col == 9:
        return True

    # If we are on the last column, we reset it and go to next row
    if col == 9:
        row += 1
        col = 0

    # If we find non-empty cell, skip it
    if grid[row][col] > 0:
        return solve_sudoku(grid, row, col + 1)

    # Here we iteratively try every number and check if it can be solved,
    # until the sudoku is solved or there is no solution
    for num in range(1, 10):
        if check_constraints(grid, row, col, num):
            grid[row][col] = num
            if solve_sudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False
