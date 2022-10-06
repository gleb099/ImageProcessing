import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('4_01.jpg')
img2 = cv2.imread('4_02.jpg')
img3 = cv2.imread('4_03.jpg')
img4 = cv2.imread('4_04.jpg')
img5 = cv2.imread('4_05.jpg')
img6 = cv2.imread('4_06.jpg')

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)
img5 = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY)
img6 = cv2.cvtColor(img6,cv2.COLOR_BGR2GRAY)

def spectr(image):
    imageSpect = np.fft.fft2(image)
    return imageSpect 

def centr(image):
    imageSpect = np.fft.fft2(image)
    fshift = np.fft.fftshift(imageSpect)
    log_shift_center = np.log(1 + np.abs(fshift))
    return log_shift_center

fig = plt.figure()

fig, ([ax1, ax2, ax3 ],[ax4, ax5, ax6] ) = plt.subplots(
    nrows=2, ncols=3,
    figsize=(18, 14)
)

ax1.set_title('1')
ax2.set_title('2')
ax3.set_title('3')
ax4.set_title('4')
ax5.set_title('5')
ax6.set_title('6')

im1 = spectr(img1)
im2 = spectr(img2)
im3 = spectr(img3)
im4 = spectr(img4)
im5 = spectr(img5)
im6 = spectr(img6)

ax1.imshow(np.log(1+np.abs(im1)), cmap='gray')
ax2.imshow(np.log(1+np.abs(im2)),cmap='gray')
ax3.imshow(np.log(1+np.abs(im3)),cmap='gray')
ax4.imshow(np.log(1+np.abs(im4)),cmap='gray')
ax5.imshow(np.log(1+np.abs(im5)),cmap='gray')
ax6.imshow(np.log(1+np.abs(im6)), cmap='gray')

plt.show()

fig = plt.figure()

fig, ([ax1, ax2, ax3 ],[ax4, ax5, ax6] ) = plt.subplots(
    nrows=2, ncols=3,
    figsize=(18, 14)
)

ax1.set_title('1')
ax2.set_title('2')
ax3.set_title('3')
ax4.set_title('4')
ax5.set_title('5')
ax6.set_title('6')

im1 = centr(img1)
im2 = centr(img2)
im3 = centr(img3)
im4 = centr(img4)
im5 = centr(img5)
im6 = centr(img6)

ax1.imshow(im1, cmap='gray')
ax2.imshow(im2,cmap='gray')
ax3.imshow(im3,cmap='gray')
ax4.imshow(im4,cmap='gray')
ax5.imshow(im5,cmap='gray')
ax6.imshow(im6, cmap='gray')

plt.show()