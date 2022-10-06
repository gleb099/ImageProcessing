import cv2
from matplotlib import pyplot as plt
from skimage import io, filters
import numpy as np

img1 = cv2.imread('test2_1.jpg')
img2 = cv2.imread('test2_2.jpg')
img3 = cv2.imread('test2_3.jpg')
img4 = cv2.imread('test2_4.jpg')
img0 = cv2.imread('test2_0.jpg')

#PSNR до фильтрации
psnrBeforeImage1 = cv2.PSNR(img0, img1)
psnrBeforeImage2 = cv2.PSNR(img0, img2)
psnrBeforeImage3 = cv2.PSNR(img0, img3)
psnrBeforeImage4 = cv2.PSNR(img0, img4)

print("PSNR для 1-го изображения, до фильтрации =", round(psnrBeforeImage1, 2))
print("PSNR для 2-го изображения, до фильтрации =", round(psnrBeforeImage2, 2))
print("PSNR для 3-го изображения, до фильтрации =", round(psnrBeforeImage3, 2))
print("PSNR для 4-го изображения, до фильтрации =", round(psnrBeforeImage4, 2))

# plt.hist(img0.ravel(), 256, [0, 256])
# plt.show()
# plt.hist(img1.ravel(), 256, [0, 256])
# plt.show()
# plt.hist(img2.ravel(), 256, [0, 256])
# plt.show()
# plt.hist(img3.ravel(), 256, [0, 256])
# plt.show()
# plt.hist(img4.ravel(), 256, [0, 256])
# plt.show()

def showImages(*args):
    fig, axes = plt.subplots(1, len(args))

    for i in range(len(axes)):
        axes[i].imshow(args[i], cmap="gray")

    fig.set_figwidth(10)    
    fig.set_figheight(10)  
    plt.show()

"""Винера"""

img1_new = cv2.medianBlur(img1, 7)
showImages(img0, img1_new)

"""Усреднений"""

img2_new = filters.gaussian(img2, 2)
img2_new = filters.unsharp_mask(filters.median(img2_new), 3)
showImages(img0, img2_new)
img2_new = np.uint8(np.around(img2_new * 255))

"""Медианный"""

# img3_new = cv2.blur(img3, (5, 5))
img3_new = cv2.medianBlur(img3, 5)
showImages(img0, img3_new)

"""Гаусса"""

img4_new  = cv2.GaussianBlur(img4, (5,5), 3)
showImages(img0, img4_new)


# img4_new = cv2.blur(img4, (5, 5))
# img4_new = cv2.medianBlur(img4_new, 3)
# showImages(img0, img4_new)

#PSNR до фильтрации
psnrBeforeImage1 = cv2.PSNR(img0, img1_new)
psnrBeforeImage2 = cv2.PSNR(img0, img2_new)
psnrBeforeImage3 = cv2.PSNR(img0, img3_new)
psnrBeforeImage4 = cv2.PSNR(img0, img4_new)

print("PSNR для 1-го изображения, после фильтрации =", round(psnrBeforeImage1, 2))
print("PSNR для 2-го изображения, после фильтрации =", round(psnrBeforeImage2, 2))
print("PSNR для 3-го изображения, после фильтрации =", round(psnrBeforeImage3, 2))
print("PSNR для 4-го изображения, после фильтрации =", round(psnrBeforeImage4, 2))