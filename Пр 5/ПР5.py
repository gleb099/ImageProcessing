from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math

image1 = cv2.imread('test1_5.tif')
image2 = cv2.imread('test2_5.tif')
img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

fft2 = np.fft.fft2(img2)

shift2center = np.fft.fftshift(fft2)

log_fft2 = np.log(1 + np.abs(fft2))

log_shift2center = np.log(1 + np.abs(shift2center))

# image1S = np.log(1+np.abs(spectr(img1)))
# image2S = np.log(1+np.abs(spectr(img2)))

plt.imshow(log_shift2center,'gray')

def process(image):
  imageSpect = np.fft.fft2(image)
  fshift = np.fft.fftshift(imageSpect)
  ifshift= np.fft.ifftshift(fshift)
  iff = np.fft.ifft2(ifshift)
  proc = (np.abs(iff))
  return proc

image1P = process(img1)
image2P = process(img2)

plt.imshow(image1P,'gray')

def distance(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def lp_filter(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base
    
def idealFilter(d, image):
    original = np.fft.fft2(image)
    center = np.fft.fftshift(original)
    LowPassCenter = center * lp_filter(d,image.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    low = np.abs(inverse_LowPass)
    return low

fig = plt.figure()

fig, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8] ) = plt.subplots(
    nrows=2, ncols=4,
    figsize=(18, 14)
)

ax1.set_title('Частота 15')
ax2.set_title('Частота 30')
ax3.set_title('Частота 50')
ax4.set_title('Частота 100')

ax5.set_title('Частота 15')
ax6.set_title('Частота 30')
ax7.set_title('Частота 50')
ax8.set_title('Частота 100')


image1S15 = idealFilter(15, img1)
image1S30 = idealFilter(30, img1)
image1S50 = idealFilter(50, img1)
image1S100 = idealFilter(100, img1)

image2S15 = idealFilter(15, img2)
image2S30 = idealFilter(30, img2)
image2S50 = idealFilter(50, img2)
image2S100 = idealFilter(100, img2)

ax1.imshow(image1S15, cmap='gray')
ax2.imshow(image1S30,cmap='gray')
ax3.imshow(image1S50,cmap='gray')
ax4.imshow(image1S100,cmap='gray')

ax5.imshow(image2S15,cmap='gray')
ax6.imshow(image2S30,cmap='gray')
ax7.imshow(image2S50, cmap='gray')
ax8.imshow(image2S100, cmap='gray')

plt.show()

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = math.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def GaussianFilter(d, image):
    original = np.fft.fft2(image)
    center = np.fft.fftshift(original)
    LowPassCenter = center * gaussianLP(d,image.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    low = np.abs(inverse_LowPass)
    return low

fig = plt.figure()

fig, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8] ) = plt.subplots(
    nrows=2, ncols=4,
    figsize=(18, 14)
)

ax1.set_title('Частота 15')
ax2.set_title('Частота 30')
ax3.set_title('Частота 50')
ax4.set_title('Частота 100')

ax5.set_title('Частота 15')
ax6.set_title('Частота 30')
ax7.set_title('Частота 50')
ax8.set_title('Частота 100')

image1G15 = GaussianFilter(15, img1)
image1G30 = GaussianFilter(30, img1)
image1G50 = GaussianFilter(50, img1)
image1G100 = GaussianFilter(100, img1)

image2G15 = GaussianFilter(15, img2)
image2G30 = GaussianFilter(30, img2)
image2G50 = GaussianFilter(50, img2)
image2G100 = GaussianFilter(100, img2)

ax1.imshow(image1G15, cmap='gray')
ax2.imshow(image1G30,cmap='gray')
ax3.imshow(image1G50,cmap='gray')
ax4.imshow(image1G100,cmap='gray')
ax5.imshow(image2G15,cmap='gray')
ax6.imshow(image2G30,cmap='gray')
ax7.imshow(image2G50, cmap='gray')
ax8.imshow(image2G100, cmap='gray')

plt.show()

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def idealLHighFilter(d, image):
    original = np.fft.fft2(image)
    center = np.fft.fftshift(original)
    HighPassCenter = center * idealFilterHP(d,image.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    high = np.abs(inverse_HighPass)
    return high

fig = plt.figure()

fig, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8] ) = plt.subplots(
    nrows=2, ncols=4,
    figsize=(18, 14)
)

ax1.set_title('Частота 15')
ax2.set_title('Частота 30')
ax3.set_title('Частота 50')
ax4.set_title('Частота 100')

ax5.set_title('Частота 15')
ax6.set_title('Частота 30')
ax7.set_title('Частота 50')
ax8.set_title('Частота 100')

image1H15 = idealLHighFilter(15, img1)
image1H30 = idealLHighFilter(30, img1)
image1H50 = idealLHighFilter(50, img1)
image1H100 = idealLHighFilter(100, img1)

image2H15 = idealLHighFilter(15, img2)
image2H30 = idealLHighFilter(30, img2)
image2H50 = idealLHighFilter(50, img2)
image2H100 = idealLHighFilter(100, img2)

ax1.imshow(image1H15, cmap='gray')
ax2.imshow(image1H30,cmap='gray')
ax3.imshow(image1H50,cmap='gray')
ax4.imshow(image1H100,cmap='gray')
ax5.imshow(image2H15,cmap='gray')
ax6.imshow(image2H30,cmap='gray')
ax7.imshow(image2H50, cmap='gray')
ax8.imshow(image2H100, cmap='gray')

plt.show()

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - math.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base
    
def GaussianHighFilter(d, image):
    original = np.fft.fft2(image)
    center = np.fft.fftshift(original)
    HighPassCenter = center * gaussianHP(d,image.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    high = np.abs(inverse_HighPass)
    return high

fig = plt.figure()

fig, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8] ) = plt.subplots(
    nrows=2, ncols=4,
    figsize=(18, 14)
)

ax1.set_title('Частота 15')
ax2.set_title('Частота 30')
ax3.set_title('Частота 50')
ax4.set_title('Частота 100')

ax5.set_title('Частота 15')
ax6.set_title('Частота 30')
ax7.set_title('Частота 50')
ax8.set_title('Частота 100')

image1GH15 = GaussianHighFilter(15, img1)
image1GH30 = GaussianHighFilter(30, img1)
image1GH50 = GaussianHighFilter(50, img1)
image1GH100 = GaussianHighFilter(100, img1)

image2GH15 = GaussianHighFilter(15, img2)
image2GH30 = GaussianHighFilter(30, img2)
image2GH50 = GaussianHighFilter(50, img2)
image2GH100 = GaussianHighFilter(100, img2)

ax1.imshow(image1GH15, cmap='gray')
ax2.imshow(image1GH30,cmap='gray')
ax3.imshow(image1GH50,cmap='gray')
ax4.imshow(image1GH100,cmap='gray')
ax5.imshow(image2GH15,cmap='gray')
ax6.imshow(image2GH30,cmap='gray')
ax7.imshow(image2GH50, cmap='gray')
ax8.imshow(image2GH100, cmap='gray')

plt.show()

