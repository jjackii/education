# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# # ! pip install opencv-python
# -

import cv2
import sys
import numpy as np

print('Hello, OpenCV', cv2.__version__)

# +
# cv2.imread(filename, flags = None) -> retval

# flags:
# cv2.IMREAD_COLOR: BGR color로 읽기 # opencv는 괜찮으나, matplotlib은 뒤집힘
# cv2.IMREAD_GRAYSCALE: 그레이 color로 읽기 Gray->Color: X (matrix값은 변함)
# cv2.IMREAD_UNCHAGED: 파일 속성대로 읽기

# retval: numpy.ndarray로 반환

# +
# filename : if, no file or permission..etc, Return empty matrix
# 확장자명 없으면 error
# flags : image read type(속성) = (default)IMREAD_COLOR
img1 = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('cat.bmp', cv2.IMREAD_COLOR)

# If can't read the img, exit
if img1 is None or img2 is None:
    print('Image load failed!')
    sys.exit()  # import sys

    # 영상의 속성 참조
print('type(img1):', type(img1), '\n')

print('img1.shape:', img1.shape) # row, column (np.array)
print('img2.shape:', img2.shape, '\n')

print('img1.dtype:', img1.dtype)
print('img1.dtype:', img2.dtype, '\n')

print('img1.shape length:', len(img1.shape))
print('img2.shape length:', len(img2.shape), '\n') # chennel(RGB)

print(img1.ndim)
print(img2.ndim)
# -

# ### img 저장

# +
# 확장자 지정 안할 시 error
cv2.imwrite('cats.png', img1, params =None) 

# filename: 저장할 파일 이름
# img: 저장할 자료 (numpy.ndarray)
# params: 파일저장 옵션
# retval: 성공 True, 실패 False

# +
# Row by Col (Height by Width)
h, w = img1.shape
print(f'img1 size: {w} x {h}')

h, w = img2.shape[:2] # ndim X
print('img2 size: {} x {}'.format(w, h))

# cv2.namedWindow('Hello OpenCV', cv2.WINDOW_NORMAL) # 창 크기 조정 가능

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.waitKey()
cv2.destroyAllWindows()
# -

# ### 영상의 픽셀값 참조

# +
# monitor 좌표 (domain 차이)
x = 20
y = 30

# matrix 좌표 
# gray
p1 = img1[y, x]
print(p1)

# color
p2 = img2[y, x]
print(p2)


img1[10:20, 10:20] = 0 # Black
img2[10:20, 10:20] = (0, 0, 255) # BGR : red

cv2.imshow('image', img1)
cv2.imshow('image2',img2)

cv2.waitKey()
cv2.destroyAllWindows()
# -

# ### 새 영상 생성

'''
numpy.empty(shape, dtype)
numpy.zeros(shape, dtype)
numpy.ones(shape, dtype)
numpy.full(shape, fill_value, dtype)
'''

# +
# 새 영상 생성하기
img1 = np.empty((240, 320), dtype=np.uint8) # Initialize X (세로, 가로)  # grayscale image
img2 = np.zeros((240, 320, 3), dtype=np.uint8) # color image
img3 = np.ones((240, 320), dtype=np.uint8) * 255 # dark gray
img4 = np.full((240, 320, 3), (0, 255, 255), dtype=np.uint8)  # yellow

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)
cv2.waitKey()
cv2.destroyAllWindows()

# +
# Copy img
img1 = cv2.imread('cat.bmp') # src

if img1 is None:
    print("image load failed")
    sys.exit()

img2 = img1 # 메모리의 같은 주소를 '참조' : python에서 주소를 가리키는 방법
img3 = img1.copy()

img1[:,:] = (0, 255, 255)

print(img1.shape)
print(img1.dtype)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)

cv2.waitKey()
cv2.destroyAllWindows()
# -

# ### 부분영상추출

# +
img1 = cv2.imread('cat.bmp')

img2 = img1[40:120, 30:150]  # numpy.ndarray의 슬라이싱
img3 = img1[40:120, 30:150].copy()

img2.fill(0)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)


while True:
    if cv2.waitKey() == 27: # ESC(ASCII) # == ord('q'):           
        break
        
cv2.destroyAllWindows()
# -


