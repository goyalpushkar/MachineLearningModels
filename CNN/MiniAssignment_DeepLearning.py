import numpy as np

def dot_product(x, w):
    sum = w[0]
    for index in range(len(x)):
        sum += x[index] * w[index+1]

    return sum


def perceptron(X, W):
    output_hat = dot_product(X, W)
    print("output_hat: ", output_hat)
    if output_hat > 0:
        return 1
    else:
        return 0

def matrix_calc_old(pixels, kernel):
    # Calculate value for each kernel size
    print("input matrix: ", pixels)
    final_value = 0
    matrix_mult = []
    for row_index in range(len(kernel)):
        row_array = []
        for col_index in range(len(kernel[0])):
            value = pixels[row_index][col_index] * kernel[row_index][col_index]
            final_value += value
            # print("Values: ", value, final_value)
            row_array.append(value)

        # print(row_array)

    matrix_mult.append(row_array)
    # print("matrix_mult: ", matrix_mult, "final_value: ", final_value)
    return matrix_mult, final_value


def matrix_calc(pixels, kernel):
    # Calculate value for each kernel size
    print("input matrix: ", pixels)
    final_value = 0
    for row_index in range(len(kernel)):
        for col_index in range(len(kernel[0])):
            final_value += pixels[row_index][col_index] * kernel[row_index][col_index]

    # print("final_value: ", final_value)
    return final_value


def convolve_helper(pixels, kernel):
    final_col_size = len(pixels[0])
    final_row_size = len(pixels)
    kernel_row_size = len(kernel)
    kernel_col_size = len(kernel[0])
    return_matrix = pixels

    if final_col_size > kernel_col_size:
        if final_row_size > kernel_row_size:

            row_size = final_row_size - kernel_row_size + 1
            col_size = final_col_size - kernel_col_size + 1

            final_matrix = []
            for row_index in range(row_size):
                row_values = []
                for col_index in range(col_size):
                    new_matrix = tuple(x[col_index:col_index+kernel_col_size] for x in pixels[row_index:row_index+kernel_row_size])
                    value = matrix_calc(new_matrix, kernel)
                    # print("calculated_matrix: ", calculated_matrix)
                    # print("value: ", value)
                    row_values.append(value)

                final_matrix.append(row_values)

            return_matrix = final_matrix

    print("return_matrix: ", return_matrix)

    final_col_size = len(return_matrix[0])
    final_row_size = len(return_matrix)
    if final_col_size > kernel_col_size:
        if final_row_size > kernel_row_size:
            convolve_helper(return_matrix, kernel)

    return return_matrix


def convolve_1(pixels, kernel):
    print("kernel: ", kernel)
    final_matrix = convolve_helper(pixels, kernel)
    result = matrix_calc(final_matrix, kernel)

    print(result)
    return result


def convolve_notworking(pixels, kernel):
    input_row_size = len(pixels)
    input_col_size = len(pixels[0])
    kernel_row_size = len(kernel)
    kernel_col_size = len(kernel[0])
    start_row = input_row_size - kernel_row_size
    start_col = input_col_size - kernel_col_size

    new_matrix = tuple(
                x[0:kernel_col_size] for x in pixels[0:kernel_row_size])
    # kernel_col_size  kernel_row_size
    value = matrix_calc(new_matrix, kernel)
    # print("calculated_matrix: ", calculated_matrix)
    # print("value: ", value)

    return value


def convolve(pixels, kernel):
    row_l = len(kernel)
    col_l = len(kernel[0])
    conv_l = 0
    for i in range(0, row_l):
        for j in range(0, col_l):
            k_i = row_l - i - 1
            k_j = col_l - j - 1
            conv_l += pixels[i][j] * kernel[k_i][k_j]

    return conv_l

# Reverse Row at specified index in the matrix
def reverseRow(data, index):
    cols = len(data[index])
    for i in range(cols // 2):
        temp = data[index][i]
        data[index][i] = data[index][cols - i - 1]
        data[index][cols - i - 1] = temp

    return data


# Print Matrix data
def printMatrix(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            print(data[i][j])

        print()


# Rotate Matrix by 180 degrees
def rotateMatrix(data):
    rows = len(data)
    cols = len(data[0])

    if (rows % 2):
        # If N is odd reverse the middle
        # row in the matrix
        data = reverseRow(data, len(data) // 2)

    # Swap the value of matrix [i][j] with
    # [rows - i - 1][cols - j - 1] for half
    # the rows size.
    for i in range(rows // 2):
        for j in range(cols):
            temp = data[i][j]
            data[i][j] = data[rows - i - 1][cols - j - 1]
            data[rows - i - 1][cols - j - 1] = temp

    return data


def reverseColumns(arr):
    for i in range(len(arr[0])):
        j = 0
        k = len(arr[0]) - 1
        while j < k:
            t = arr[j][i]
            arr[j][i] = arr[k][i]
            arr[k][i] = t
            j += 1
            k -= 1

# Function for transpose of matrix
def transpose(arr):
    for i in range(len(arr)):
        for j in range(i, len(arr[0])):
            t = arr[i][j]
            arr[i][j] = arr[j][i]
            arr[j][i] = t


def rotate180(arr):
    transpose(arr);
    reverseColumns(arr);
    transpose(arr);
    reverseColumns(arr);
    return arr


def dot_product_np(pixels, kernel):
    matrix_mult = np.dot(pixels, kernel)
    return matrix_mult


if __name__ == '__main__':
    X = [1, -2, 3, -5]
    W = [10, 1, 2, 3, -5]

    output = perceptron(X, W)
    print("output: ", output)


    kernel = ([0, -1, 0], [-1, 5, -1], [0, -1, 0])
    # kernel = [[0, -1, 1], [-1, 5, 2], [1, 0, 1], [1, 2, 2]]

    matrix1 = ([105, 102, 100], [103, 99, 103], [101, 98, 104])

    matrix3 = ([100, 101, 102, 103, 104], [200, 201, 202, 203, 204],
               [300, 301, 302, 303, 304], [400, 401, 402, 403, 404],
               [500, 501, 502, 503, 504])

    pixels = [[5, 6, 7], [2, 3, 1], [1, 1, 1]]

    # print( tuple(x[0:3] for x in matrix1[0:3]) )
    kernel_rotated = rotateMatrix(kernel)
    print("kernel_rotated: ", kernel_rotated)
    final_value = convolve(matrix1, kernel_rotated)
    print(final_value)  #  [1]

