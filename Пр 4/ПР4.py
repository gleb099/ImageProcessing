import math
from scipy import signal, ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import util, filters

x = np.linspace(0, 1, 500, endpoint=False)
plt.plot(x, signal.square(2 * np.pi * 5 * x))
plt.ylim(-2, 2)

def function():
    return (signal.square(2 * np.pi * 5 * x))
b = function()

def image(p):
    p = p * 255
    image = np.full((len(p), len(p)), fill_value=p)
    plt.imshow(image, cmap="gray")
    plt.savefig('image.jpg')
    return image
a = image(b)

#линейный сглаживающий фильтр
lineImage = cv2.blur(a, (10, 10))
plt.imshow(lineImage, cmap='gray')

#импульсный шум
noiseImage = util.random_noise(a, "pepper", seed=3)
plt.imshow(noiseImage, cmap='gray')

#медианная фильтрация
medianImage = filters.median(noiseImage, mode="nearest")
plt.imshow(medianImage, cmap='gray')

#повышение резкости
lapImage = np.array(lineImage.copy(), dtype=np.uint8)
lapImage = cv2.Laplacian(lapImage, cv2.CV_16S, ksize=13)
plt.imshow(lapImage, cmap='gray')

def laplacian_sharpening(img, K_size=3):
 H, W = img.shape
 # zero padding
 pad = K_size // 2
 
 out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
 
 out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
 
 tmp = out.copy()
 
 # laplacian kernle
 
 K = [[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]]
 
 # filtering and adding image -> Sharpening image
 
 for y in range(H):
  for x in range(W):
   # core code
    out[pad + y, pad + x] = (-1) * np.sum(K * (tmp[y: y + K_size, x: x + K_size])) + tmp[pad + y, pad + x]
 out = np.clip(out, 0, 255)
 out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
 
 return out
 
# Read Gray Scale image
 
img = lineImage.copy().astype(np.float)
 
# Image sharpening by laplacian filter
 
out = laplacian_sharpening(img, K_size=3)

plt.imshow(out, cmap='gray')