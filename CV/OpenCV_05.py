# 객체 추적
## 01

import sys
import numpy as np
import cv2


# 비디오 파일 열기
cap = cv2.VideoCapture('PETS2000.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 영상 등록
ret, back = cap.read()

if not ret:
    print('Background image registration failed!')
    sys.exit()

back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY) # subtraction 할거라서 color필요 x
# back = cv2.GaussianBlur(back, (0, 0), 1.0)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frame읽어서 to gray
    # gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # 차영상 구하기 & 이진화
    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()



## 02
# 정적 배경 차분 (맨 처음 시점/고정 - 현재)

# 비디오 파일 열기
cap = cv2.VideoCapture('PETS2000.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 영상 등록
ret, back = cap.read()

if not ret:
    print('Background image registration failed!')
    sys.exit()

back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0, 0), 1.0)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # 차영상 구하기 & 이진화
    diff = cv2.absdiff(gray, back) # 계속 subtraction -> 움직이는 부분만 나옴
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY) # 30~255 binary로

    # 레이블링을 이용하여 바운딩 박스 표시
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff) # 위치정보를 반환해줌

    for i in range(1, cnt):
        x, y, w, h, s = stats[i]

        if s < 100:
            continue

        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2) # 빨간색, 두께

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# 이동평균 배경차분
# 비디오 파일 열기
cap = cv2.VideoCapture('PETS2000.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 영상 등록
ret, back = cap.read()

if not ret:
    print('Background image registration failed!')
    sys.exit()

# back: uint8 배경, fback: float32 배경
back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY) # 받을 때 8bit
back = cv2.GaussianBlur(back, (0, 0), 1.0)
fback = back.astype(np.float32) # 입력 32 (연산)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # fback: float32, back: uint8 배경
    cv2.accumulateWeighted(gray, fback, 0.01) # 현재 영상 0.01씩? update -> 이전 영상 0.9 # 1주면 박스 안나옴
    back = fback.astype(np.uint8) # 출력 다시 8bit(255)

    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 레이블링을 이용하여 바운딩 박스 표시
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)

    for i in range(1, cnt):
        x, y, w, h, s = stats[i]

        if s < 100:
            continue

        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)
    cv2.imshow('back', back)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()