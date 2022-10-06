from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import math as m
import cv2
import matplotlib.pylab as pl
from skimage.exposure import histogram
import math

image = cv2.imread('test.jpg')

plt.hist(image.ravel(), 256, [0, 256])
plt.show()

print("Максимальное значение светлоты:\t", np.max(image))
print("Минимальное значение светлоты:\t", np.min(image))
print("Контрастность:\t", np.max(image) - np.min(image))

"""Логарифмическое преобразование"""

def log_plot(c):
  x = np.arange(0, 256, 0.01)
  y = c * np.log(1 + x)

def log(c, image2):
  output = c * np.log(1.0 + image2)
  output = np.uint8(output + 0.5)
  return output

log_plot(32)
output = log(32, image)

log_transformed = np.array(output, dtype = np.uint8)
plt.hist(output.ravel(),bins = 256, histtype = 'step' )

cv2.imwrite('логарифмическое преобразование.jpg', log_transformed)

print("Максимальное значение светлоты:\t", np.max(log_transformed))
print("Минимальное значение светлоты:\t", np.min(log_transformed))
print("Контрастность:\t", np.max(log_transformed) - np.min(log_transformed))



"""Кусочно линейное преобразование"""

img = image.copy()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r_min, r_max = 255, 0
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        if gray_img[i, j] > r_max:
            r_max = gray_img[i, j]
        if gray_img[i, j] < r_min:
            r_min = gray_img[i, j]
r1, s1 = r_min, 0
r2, s2 = r_max, 255

precewise_img = np.zeros((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
k1 = s1 / r1
k3 = (255 - s2) / (255 - r2)
k2 = (s2 - s1) / (r2 - r1)
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        if r1 <= gray_img[i, j] <= r2:
            precewise_img[i, j] = k2 * (gray_img[i, j] - r1)
        elif gray_img[i, j] < r1:
            precewise_img[i, j] = k1 * gray_img[i, j]
        elif gray_img[i, j] > r2:
            precewise_img[i, j] = k3 * (gray_img[i, j] - r2)

cv2.imwrite('кусочно-линейное преобразование.jpg', precewise_img)
plt.hist(precewise_img.ravel(), bins=256, rwidth=0.8, range=(0, 255))
plt.show()

print("Максимальное значение светлоты:\t", np.max(precewise_img))
print("Минимальное значение светлоты:\t", np.min(precewise_img))
print("Контрастность:\t", np.max(precewise_img) - np.min(precewise_img))

"""Степенное преобразование"""

def deg(img, pow, c = 1):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for p in range(len(img[i][j])):
                img[i][j][p] = 255 * (img[i][j][p]/255) **pow
img = image.copy()
deg(img, 1.5)

cv2.imwrite('степенное преобразование.jpg', img)

plt.hist(img.ravel(), bins=256, rwidth=0.8, range=(0, 255))
plt.show()

print("Максимальное значение светлоты:\t", np.max(img))
print("Минимальное значение светлоты:\t", np.min(img))
print("Контрастность:\t", np.max(img) - np.min(img))

"""Эквализация"""

img = image.copy()
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# apply histogram equalization 
out = cv2.equalizeHist(img)
cv2.imwrite('эквализация.jpg', out)
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()

print("Максимальное значение светлоты:\t", np.max(out))
print("Минимальное значение светлоты:\t", np.min(out))
print("Контрастность:\t", np.max(out) - np.min(out))

"""Нормализация гистограммы"""

img = image.copy()
out = cv2.normalize(img, None, -50, 400, norm_type=cv2.NORM_MINMAX)

cv2.imwrite('нормализация гистограммы.jpg', out)
plt.hist(out.ravel(), 256, [0, 256])
plt.show()

print("Максимальное значение светлоты:\t", np.max(out))
print("Минимальное значение светлоты:\t", np.min(out))
print("Контрастность:\t", np.max(out) - np.min(out))