import cv2
import numpy as np


def salt_pepper_noise(image, fraction):
    img = np.copy(image)
    row, column = img.shape
    w = 0
    b = 0
    for i in range(0, row):
        for j in range(0, column):
            if (np.random.randint(100) < fraction):
                img[i, j] = 255
                w = w + 1
            if (np.random.randint(100) < fraction):
                img[i, j] = 0
                b = b + 1

    print(w)
    print(b)
    return img


# # # # # # # # # # #
if __name__ == '__main__': img = cv2.imread('imgData/lena_std.jpg', cv2.IMREAD_GRAYSCALE)
fraction = 25
noisy = salt_pepper_noise(img, fraction)
cv2.imshow('Salt & Pepper Noise', noisy)

cv2.waitKey(0)
cv2.destroyAllWindows()
