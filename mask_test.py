# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:17:00 2020

@author: DELL
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 


# =============================================================================
# def add_mask2image_binary(images_path, masks_path, masked_path):
# # Add binary masks to images
#   for img_item in os.listdir(images_path):
#     #print(img_item)
#     img_path = os.path.join(images_path, img_item)
#     img = cv2.imread(img_path)
#     mask_path = os.path.join(masks_path, img_item) #[:-4]+'.jpg') # mask是.png格式的，image是.jpg格式的
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 将彩色mask以二值图像形式读取
#     cv2.imwrite(os.path.join(masks_path, img_item), mask) #输出二值图像的mask
#     masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask) #将image的相素值和mask像素值相加得到结果
#     cv2.imwrite(os.path.join(masked_path, img_item), masked)
# images_path = 'E:\python\mask_test\image_path'
# masks_path = 'E:\python\mask_test\masks_path'
# masked_path = 'E:\python\mask_test\masked_path'
# add_mask2image_binary(images_path, masks_path, masked_path)
# 
# =============================================================================
 
# img=cv2.imread('E:\python\mask_test\masks_path\SEQ_10_001_5_2_6302_1558_7861_3117.jpg')
 
# print(img.shape())
 
# print(img)


#=================提取图像的蓝色部分==============
 
#读入的图像是BGR空间图像
#最原始的图片
frame_org = cv2.imread("E:\instance_seg_input_image\SEQ_10_001_3_44_5651_14300_6302_14951.jpg")
#实例分割后的图片
frame = cv2.imread("E:\instance_seg_output_image\SEQ_10_001_3_44_5651_14300_6302_14951.jpg")
 
# 部分1：将BGR空间的图片转换到HSV空间
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
#部分2：
# 在HSV空间中定义蓝色
lower_blue = np.array([100,43,46])
upper_blue = np.array([124, 255, 255])
# 在HSV空间中定义橙色
lower_orange = np.array([26,43,46])
upper_orange = np.array([34, 255, 255])

 
#部分3：
# 从HSV图像中截取出蓝色、橙色，即获得相应的掩膜
# cv2.inRange()函数是设置阈值去除背景部分，得到想要的区域
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
 
#部分4：
# 将原图像和mask(掩膜)进行按位与
blue_res = cv2.bitwise_and(frame_org, frame_org, mask = blue_mask)
orange_res = cv2.bitwise_and(frame_org, frame_org, mask = orange_mask)

masked_blue = cv2.add(frame_org, np.zeros(np.shape(frame_org), dtype=np.uint8), mask=blue_mask) #将image的相素值和mask像素值相加得到结果
 
#最后得到要分离出的颜色图像
# res = blue_res + green_res + red_res
 
 
#部分5:将BGR空间下的图片转换成RGB空间下的图片
frame_org = frame_org[:,:,::-1]
frame = frame[:,:,::-1]
blue_res = blue_res[:,:,::-1]
orange_res = orange_res[:,:,::-1]
masked_blue=masked_blue[:,:,::-1]
 
#部分6：显示图像 mask
plt.figure(figsize=(14,12))
plt.subplot(2,2,1),plt.title('original_image'), plt.imshow(frame_org)
plt.subplot(2,2,2), plt.imshow(blue_mask, cmap = 'gray')
plt.subplot(2,2,3), plt.imshow(orange_mask, cmap= 'gray')
# plt.subplot(2,2,4), plt.imshow(red_mask, cmap= 'gray')
 
#显示mask之后的图像
plt.figure(figsize=(14,12))
#plt.subplot(2,2,1), plt.imshow(blue_res)
plt.subplot(2,2,1), plt.imshow(masked_blue)
#plt.subplot(2,2,2), plt.imshow(orange_res)
# plt.subplot(2,2,3), plt.imshow(red_res)
# plt.subplot(2,2,4), plt.imshow(res)
plt.show()

#部分7：存储RGB空间下的照片
#cv2.imwrite('./dog_split'+str(ii)+'.png', im2col[ii]*255)

# cv2.imwrite('./blue.png', blue_res)
cv2.imwrite('./mask_blue.png', blue_mask)
cv2.imwrite('./masked_and_blue.png', blue_res) #两个效果一样
cv2.imwrite('./masked_add_blue.png', masked_blue)
# cv2.imwrite('./orange.png', orange_res)

