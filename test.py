import cv2
import numpy as np


def salt_pepper_noise(image, fraction, salt_vs_pepper):
    img = np.copy(image)
    size = img.size
    num_salt = np.ceil(fraction * size * salt_vs_pepper).astype('int')
    num_pepper = np.ceil(fraction * size * (1 - salt_vs_pepper)).astype('int')
    row, column = img.shape

    # 隨機的座標點
    x = np.random.randint(0, column - 1, num_pepper)
    y = np.random.randint(0, row - 1, num_pepper)
    img[y, x] = 0  # 撒上胡椒

    # 隨機的座標點
    x = np.random.randint(0, column - 1, num_salt)
    y = np.random.randint(0, row - 1, num_salt)
    img[y, x] = 255  # 撒上鹽
    return img


fraction = 0.25  # 雜訊佔圖的比例
salt_vs_pepper = 0.5  # 鹽與胡椒的比例

img = cv2.imread('imgData/lena_std.jpg', cv2.IMREAD_GRAYSCALE)
noisy = salt_pepper_noise(img, fraction, salt_vs_pepper)

cv2.imshow('Salt & Pepper Noise', noisy)

cv2.waitKey(0)
cv2.destroyAllWindows()
