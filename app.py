import cv2
import numpy as np
import imageNoiseRemoval


# 添加胡椒鹽雜訊
def salt_pepper_noise(image, fraction):
    print('Add Salt & Pepper Noise running...')
    img = np.copy(image)
    row, column = img.shape
    w = 0
    b = 0
    for i in range(0, row):
        for j in range(0, column):
            random_number = np.random.randint(100)
            if random_number < fraction:
                img[i, j] = 255
                w = w + 1
            elif random_number >= fraction and random_number < fraction * 2:
                img[i, j] = 0
                b = b + 1

    print('Add Salt & Pepper Noise done!')
    print('Salt : ' + str(w))
    print('Pepper : ' + str(b))
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
    edge_x_ed = x + s_xy + 1 if x + s_xy + 1 < img.shape[0] else img.shape[0]
    edge_y_ed = y + s_xy + 1 if y + s_xy + 1 < img.shape[1] else img.shape[1]

    filter_window = img[edge_x_st: edge_x_ed, edge_y_st: edge_y_ed]

    target = filter_window.reshape(-1)
    z_xy = img[x, y]
    z_min = np.min(target)
    z_max = np.max(target)
    z_med = calculate_median(target)

    if z_min < z_med < z_max:
        return amf_level_b(z_min, z_med, z_max, z_xy)

    else:
        s_xy += 1
        if (s_xy <= s_max):
            return amf_level_a(img, x, y, s_xy)
        else:
            return z_med


# B部分
def amf_level_b(z_min, z_med, z_max, z_xy):
    if z_min < z_xy < z_max:
        return z_xy
    else:
        return z_med


# median filter
def median_filter(img, x, y, s_xy=1):
    edge_x_st = x - s_xy if x >= s_xy else 0
    edge_y_st = y - s_xy if y >= s_xy else 0
    edge_x_ed = x + s_xy + 1 if x + s_xy + 1 < img.shape[0] else img.shape[0]
    edge_y_ed = y + s_xy + 1 if y + s_xy + 1 < img.shape[1] else img.shape[1]

    filter_window = img[edge_x_st: edge_x_ed, edge_y_st: edge_y_ed]
    target = filter_window.reshape(-1)
    median_value = calculate_median(target)

    return median_value


# # # # # # # # # # #
if __name__ == '__main__':
    # imageNoiseRemoval.image_noise_removal()
    #
    #
    # import os
    # bad_frames = 'imgData/train'
    # noisy_frames = []
    # fraction = 25
    # i = 0
    # for file in os.listdir(bad_frames):
    #     if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
    #         img_o = cv2.imread('imgData/train/' + file, cv2.IMREAD_GRAYSCALE)
    #         img_o = cv2.resize(img_o, (256, 256), interpolation=cv2.INTER_AREA)
    #         if file == 'zzzz9999.jpg':
    #             cv2.imwrite('imgData/good_frames/99999.jpg', img_o)
    #         else:
    #             cv2.imwrite('imgData/good_frames/' + str(i) + '.jpg', img_o)
    #         i = i + 1
    #
    #
    #
    #
    # bad_frames = 'imgData/good_frames'
    # noisy_frames = []
    # fraction = 25
    # for file in os.listdir(bad_frames):
    #     if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
    #         img_o = cv2.imread('imgData/good_frames/' + file, cv2.IMREAD_GRAYSCALE)
    #         img_o = cv2.resize(img_o, (256, 256), interpolation=cv2.INTER_AREA)
    #         noisy = salt_pepper_noise(img_o, fraction)
    #         cv2.imwrite('imgData/bad_frames/' + file, noisy)
    #

    img_o = cv2.imread('imgData/lena_std.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img_o', img_o)
    fraction = 25
    noisy = salt_pepper_noise(img_o, fraction)
    amf_out_img = np.empty([img_o.shape[0], img_o.shape[1]])
    median_out_img = np.empty([img_o.shape[0], img_o.shape[1]])
    cv2.imshow('Salt & Pepper Noise', noisy)

    # Filter Start
    print('Adaptive Median Filter running...')
    x_length, y_length = img_o.shape
    for i in range(0, x_length):
        for j in range(0, y_length):
            amf_out_img[i][j] = amf_level_a(noisy, i, j) / 255  # Adaptive Median Filter
            median_out_img[i][j] = median_filter(noisy, i, j) / 255  # Median Filter

    print('Adaptive Median Filter done!')
    cv2.imshow('amf_out_img', amf_out_img)
    cv2.imshow('median_out_img', median_out_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
