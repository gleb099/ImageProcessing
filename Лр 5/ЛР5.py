import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image, ImageDraw
from scipy import ndimage

img0 = cv2.imread('test5_0.jpg')
img1 = cv2.imread('test5_1.jpg')
img2 = cv2.imread('test5_2.jpg')
img3 = cv2.imread('test5_3.jpg')
img4 = cv2.imread('test5_4.jpg')
img5 = cv2.imread('test5_5.jpg')
# img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

def contourArea(im):
  ret, thresh = cv2.threshold(im,1,255,0)
  contours, hierarchy = cv2.findContours(thresh, 1,2)
  cnt = contours[0]
  return cv2.contourArea(cnt)

def drawImage(image, title, h):
    axes[h].set_title(title)
    axes[h].imshow(image, cmap='gray')

def roberts(img):
  kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
  kernely = np.array([[0, -1], [1, 0]], dtype=int)
  x = cv2.filter2D(img, cv2.CV_16S, kernelx)
  y = cv2.filter2D(img, cv2.CV_16S, kernely)
  absX = cv2.convertScaleAbs(x)
  absY = cv2.convertScaleAbs(y)
  Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
  return  Roberts

img0R = roberts(img0)
img1R = roberts(img1)
img2R = roberts(img2)
img3R = roberts(img3)
img4R = roberts(img4)
img5R = roberts(img5)

fig, axes = plt.subplots(1,6)
fig.set_figwidth(30) 
fig.set_figheight(20)

drawImage(img0R, 'Оператор Робертса №1',0)
drawImage(img1R, 'Оператор Робертса №2',1)
drawImage(img2R, 'Оператор Робертса №3',2)
drawImage(img3R, 'Оператор Робертса №4',3)
drawImage(img4R, 'Оператор Робертса №5',4)
drawImage(img5R, 'Оператор Робертса №6',5)


arr=[]
for i in [img1R, img2R, img3R, img4R, img5R, img0R]:
    x = contourArea(i)
    arr.append( x)
print(arr)

def prewitt(im):
  kernelx = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=int)
  kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
  x = cv2.filter2D(im, cv2.CV_16S, kernelx)
  y = cv2.filter2D(im, cv2.CV_16S, kernely)
  absX = cv2.convertScaleAbs(x)
  absY = cv2.convertScaleAbs(y)
  Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
  return Prewitt

img0P = prewitt(img0)
img1P = prewitt(img1)
img2P = prewitt(img2)
img3P = prewitt(img3)
img4P = prewitt(img4)
img5P = prewitt(img5)

fig, axes = plt.subplots(1,6)
fig.set_figwidth(30) 
fig.set_figheight(20)

drawImage(img0P, 'Оператор Превитта №1',0)
drawImage(img1P, 'Оператор Превитта №2',1)
drawImage(img2P, 'Оператор Превитта №3',2)
drawImage(img3P, 'Оператор Превитта №4',3)
drawImage(img4P, 'Оператор Превитта №5',4)
drawImage(img5P, 'Оператор Превитта №6',5)

def sobel(im):
  kernelx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=int)
  kernely = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=int)
  x = cv2.filter2D(im, cv2.CV_16S, kernelx)
  y = cv2.filter2D(im, cv2.CV_16S, kernely)
  absX = cv2.convertScaleAbs(x)
  absY = cv2.convertScaleAbs(y)
  Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
  return Sobel

img0S = sobel(img0)
img1S = sobel(img1)
img2S = sobel(img2)
img3S = sobel(img3)
img4S = sobel(img4)
img5S = sobel(img5)

fig, axes = plt.subplots(1,6)
fig.set_figwidth(30) 
fig.set_figheight(20)

drawImage(img0S, 'Оператор Собела №1',0)
drawImage(img1S, 'Оператор Собела №2',1)
drawImage(img2S, 'Оператор Собела №3',2)
drawImage(img3S, 'Оператор Собела №4',3)
drawImage(img4S, 'Оператор Собела №5',4)
drawImage(img5S, 'Оператор Собела №6',5)

def laplacian(im):
  dst = cv2.Laplacian(im, cv2.CV_16S, ksize = 3)
  Laplacian = cv2.convertScaleAbs(dst)
  return Laplacian

img0L = laplacian(img0)
img1L = laplacian(img1)
img2L = laplacian(img2)
img3L = laplacian(img3)
img4L = laplacian(img4)
img5L = laplacian(img5)

fig, axes = plt.subplots(1,6)
fig.set_figwidth(30) 
fig.set_figheight(20)

drawImage(img0L, 'Оператор Лапласина №1',0)
drawImage(img1L, 'Оператор Лапласина №2',1)
drawImage(img2L, 'Оператор Лапласина №3',2)
drawImage(img3L, 'Оператор Лапласина №4',3)
drawImage(img4L, 'Оператор Лапласина №5',4)
drawImage(img5L, 'Оператор Лапласина №6',5)