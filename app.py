import cv2
import numpy as np


# 添加胡椒鹽雜訊
def salt_pepper_noise(image, fraction):
    img = np.copy(image)
    row, column = img.shape
    w = 0
    b = 0
    for i in range(0, row):
        for j in range(0, column):
            if np.random.randint(100) < fraction:
                img[i, j] = 255
                w = w + 1
            if np.random.randint(100) < fraction:
                img[i, j] = 0
                b = b + 1

    print(w)
    print(b)
    return img


# 找出中值
def calculate_median(array):
    sorted_array = np.sort(array)
    median = sorted_array[len(array) // 2]
    return median


# A部分
def amf_level_a(img, x, y, s_xy=1, s_max=3):
    edge_x_st = x - s_xy if x >= s_xy else 0
    edge_y_st = y - s_xy if y >= s_xy else 0
    edge_x_ed = s_xy * 2 + 1 if s_xy * 2 + 1 <= img[0].shape else img[0].shape
    edge_y_ed = s_xy * 2 + 1 if s_xy * 2 + 1 <= img[0].shape else img[1].shape

    filter_window = img[edge_x_st: edge_x_ed, edge_y_st: edge_y_ed]

    target = filter_window.reshape(-1)
    z_xy = img[x, y]
    z_min = np.min(target)
    z_max = np.max(target)
    z_med = calculate_median(target)

    return 0


# # # # # # # # # # #
if __name__ == '__main__':
    img_o = cv2.imread('imgData/lena_std.jpg', cv2.IMREAD_GRAYSCALE)
    fraction = 25
    noisy = salt_pepper_noise(img_o, fraction)

    # Adaptive Median Filter Start
    x_length, y_length = img_o.shape
    for i in range(0, x_length):
        for j in range(0, y_length):
            amf_level_a(img_o, i, j)

    cv2.imshow('Salt & Pepper Noise', noisy)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
