from matplotlib import pyplot as plt
import numpy as np
import cv2
from skimage.exposure import histogram
import os
import math

image1 = cv2.imread('01.jpg')
image2 = cv2.imread('02.jpg')

plt.hist(image1.ravel(), bins=256, histtype='step')
plt.show()

#нормализация
img = image2.copy()
out = cv2.normalize(img, None, -50, 400, norm_type=cv2.NORM_MINMAX)

cv2.imwrite('02_norm.jpg', out)
plt.hist(out.ravel(), bins=256, rwidth=0.8, range=(0, 255))
plt.show()

#эквализация

img = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

# apply histogram equalization 
out = cv2.equalizeHist(img)
cv2.imwrite('02_ecv.jpg', out)
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()

x = np.arange(1, 255, 1)
y = 5 * x + 5

def find_closest(A, target):
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return A[idx]
def function_prived(image, x, y):
    # функция распределения гистограммы
    hist, hist_centers = histogram(image)
    img_accum = []
    func_accum = []    
    dict_img = {}
    dict_func = {}
    dict_res = {}
    s = 0
    for i in range(len(hist)):
        s += hist[i]
        img_accum.append(s)
    for key, val in zip(hist_centers, img_accum):
        dict_img[key] = val
    s = 0
    for i in range(len(y)):
        s += y[i]
        func_accum.append(s)
    for key, val in zip(x, func_accum):
        dict_func[key] = val
    A = np.array(func_accum)
    target = list(img_accum)
    new_match = find_closest(A, target)
    tmp = []
    for i in find_closest(A, target):
        for key, value in dict_func.items():  
            if value == i:
                tmp.append(key)
    for key, value in zip(hist_centers, tmp):
        dict_res[key] = value
    shape_img = image.shape   
    image_copy = image.copy()
    image_copy = image_copy.ravel()
    for i in range(len(image_copy)):
        image_copy[i] = dict_res[image_copy[i]]
    image_copy = image_copy.reshape(shape_img)   
    return image_copy   

out = function_prived(image2, x, y)

cv2.imwrite('02_fun.jpg', out)
plt.hist(out.ravel(), bins=256, range=(0, 253))
plt.show()

x = np.arange(1, 255, 1)

fig, ax = plt.subplots()
ax.plot(x, y, color="red")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

plt.show()