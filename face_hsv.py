import cv2, dlib, math, matplotlib
import numpy as np

# 슬라이더 Callback 함수
def onChange(pos):
    pass

# 이미지 불러오기
img = cv2.imread('./image/8.jpg')

# Haar-Cascade classifier를 이용하여 얼굴 부분 감지
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)

# 얼굴 영역을 ROI로 설정
for k, d in enumerate(dets): 
    shape = predictor(img, d)
    img_roi = img[d.top():d.bottom(), d.left():d.right()]

# 얼굴 영역을 HSV 색공간으로 변환 후 3개의 채널로 나눔
img_hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(img_hsv)

# 슬라이더 생성
# lambda : 아무 행동 작업도 하지 않음
cv2.namedWindow("Slider Window")
cv2.createTrackbar("H_th", "Slider Window", 0, 255, onChange)
cv2.createTrackbar("H_max", "Slider Window", 0, 255, lambda x: x)
cv2.createTrackbar("S_th", "Slider Window", 0, 255, onChange)
cv2.createTrackbar("S_max", "Slider Window", 0, 255, lambda x: x)
cv2.createTrackbar("V_th", "Slider Window", 0, 255, onChange)
cv2.createTrackbar("V_max", "Slider Window", 0, 255, lambda x: x)

# 슬라이더 초기화
cv2.setTrackbarPos("H_th", "Slider Window", 0)
cv2.setTrackbarPos("H_max", "Slider Window", 255)
cv2.setTrackbarPos("S_th", "Slider Window", 0)
cv2.setTrackbarPos("S_max", "Slider Window", 255)
cv2.setTrackbarPos("V_th", "Slider Window", 0)
cv2.setTrackbarPos("V_max", "Slider Window", 255)

# 슬라이더 작동
# q를 누르면 모든 윈도우 창 종료
while cv2.waitKey(1) != ord('q'):

    # 슬라이더 값 적용
    H_th = cv2.getTrackbarPos("H_th", "Slider Window")
    H_max = cv2.getTrackbarPos("H_max", "Slider Window")
    S_th = cv2.getTrackbarPos("S_th", "Slider Window")
    S_max = cv2.getTrackbarPos("S_max", "Slider Window")
    V_th = cv2.getTrackbarPos("V_th", "Slider Window")
    V_max = cv2.getTrackbarPos("V_max", "Slider Window")

    # 적용된 슬라이더 값으로 하한, 상한 값 Tuple 지정
    lower = (H_th, S_th, V_th)
    upper = (H_max, S_max, V_max)

    # 하한, 상한 범위 내의 픽셀을 Image Mask로 지정
    img_mask = cv2.inRange(img_hsv, lower, upper)

    # 비트연산을 위해 이미지 배열 크기 변경
    img_merge = cv2.merge((img_mask, img_mask, img_mask))

    # 원본 얼굴 사진과 마스크에 해당하는 부분 AND 연산 -> 피부 이미지
    img_skin = cv2.bitwise_and(img_roi, img_merge)

    # 원본 얼굴 이미지 출력
    cv2.imshow("Original", img_mask)
    # 피부 이미지 출력
    cv2.imshow("Skin", img_skin)

cv2.destroyAllWindows()