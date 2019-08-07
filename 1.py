
import matplotlib.pyplot as plt
import numpy as np
import cv2

im_array = cv2.imread("285.jpg") # opencv返回BGR
im_array1 = im_array[:,:,[2,1,0]]  # plt按RGB显示，逆转一下
figure = plt.figure()

plt.subplot(121)
plt.imshow(im_array)

rect = plt.Rectangle((50, 50), # 左上角坐标点
                     250,      # 宽
                     250,      # 高
                     fill=False,
                     edgecolor='red', linewidth=0.7)
plt.gca().add_patch(rect)

plt.scatter(50,50,c='yellow',linewidths=0.1, marker='o', s=50)
plt.scatter(50,300,c='yellow',linewidths=1, marker='x', s=5)
plt.scatter(300,50,c='yellow',linewidths=0.1, marker='x', s=5)
plt.scatter(300,300,c='yellow',linewidths=0.1, marker='x', s=5)


plt.subplot(122)
plt.imshow(im_array1)
rect = plt.Rectangle((50, 50), # 左上角坐标点
                     250,      # 宽
                     250,      # 高
                     fill=False,
                     edgecolor='red', linewidth=0.7)
plt.gca().add_patch(rect)


plt.axis('off')
plt.savefig("save_name")
plt.show()
