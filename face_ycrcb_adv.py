import cv2, dlib, math, matplotlib
import numpy as np

# 슬라이더 Callback 함수
def onChange(pos):
    pass

# 슬라이더 생성
# lambda : 아무 행동 작업도 하지 않음
cv2.namedWindow("Slider Window")
cv2.createTrackbar("Y_th", "Slider Window", 0, 255, onChange)
cv2.createTrackbar("Y_max", "Slider Window", 0, 255, onChange)
cv2.createTrackbar("Cr_th", "Slider Window", 0, 255, onChange)
cv2.createTrackbar("Cr_max", "Slider Window", 0, 255, onChange)
cv2.createTrackbar("Cb_th", "Slider Window", 0, 255, onChange)
cv2.createTrackbar("Cb_max", "Slider Window", 0, 255, onChange)

# 이미지 불러오기
img = cv2.imread('./image/20.jpg')

# Haar-Cascade classifier를 이용하여 얼굴 부분 감지
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)

for k, d in enumerate(dets): 
    shape = predictor(img, d)

    # 눈, 입의 특징점 배열
    eyebrow = np.empty((0,2), np.int32)
    left_eye = np.empty((0,2), np.int32)
    right_eye = np.empty((0,2), np.int32)
    mouth = np.empty((0,2), np.int32)

    for i in range(shape.num_parts):
        shape_point = shape.part(i)

        # 왼쪽 눈 좌표 지정
        if i >= 36 and i <= 41:
            # cv2.circle(img, (shape_point.x, shape_point.y), circle_r, color_l_out, line_width)
            left_eye = np.append(left_eye, np.array([[shape_point.x, shape_point.y]]), axis=0)
        
        # 오른쪽 눈 좌표 지정
        if i >= 42 and i <= 47:
            # cv2.circle(img, (shape_point.x, shape_point.y), circle_r, color_l_out, line_width)
            right_eye = np.append(right_eye, np.array([[shape_point.x, shape_point.y]]), axis=0)

        # 입 좌표 지정
        if i >= 48 and i <= 59:
            # cv2.circle(img, (shape_point.x, shape_point.y), circle_r, color_l_out, line_width)
            mouth = np.append(mouth, np.array([[shape_point.x, shape_point.y]]), axis=0)

    # 눈, 입 특징 좌표 출력 (디버그용)
    # print('Left eye : ', left_eye)
    # print('Right eye : ', right_eye)
    # print('Mouth : ', mouth)

    # 눈, 입 특징 좌표를 포함하는 타원 탐색
    left_eye_ellipse = cv2.fitEllipse(left_eye)
    right_eye_ellipse = cv2.fitEllipse(right_eye)
    mouth_ellipse = cv2.fitEllipse(mouth)

    # 타원을 화면에 도시
    cv2.ellipse(img, left_eye_ellipse, (0,0,0), -1)
    cv2.ellipse(img, right_eye_ellipse, (0,0,0), -1)
    cv2.ellipse(img, mouth_ellipse, (0,0,0), -1)

    img_top = d.top()
    img_bottom = d.bottom()
    img_left = d.left()
    img_right = d.right()

# 얼굴 영역을 HSV 색공간으로 변환 후 3개의 채널로 나눔
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(img_ycrcb)

# # Otsu 알고리즘을 이용하여 적절한 하한, 상한 값 Tuple 지정
# ret_S, mat_S = cv2.threshold(S, -1, 255, cv2.THRESH_OTSU)
# ret_V, mat_V = cv2.threshold(V, -1, 255, cv2.THRESH_OTSU)
# Cr_th = ret_S
# Cb_th = ret_V
# Cr_max = Cb_max = 255

# # Otsu threshold value (디버그용)
# print('S thres value : ', ret_S)
# print('V thres value : ', ret_V)

# # 가중치
# S_val = 0.4
# V_val = 1.3

# 슬라이더 초기화
cv2.setTrackbarPos("Y_th", "Slider Window", 0)
cv2.setTrackbarPos("Y_max", "Slider Window", 255)
cv2.setTrackbarPos("Cr_th", "Slider Window", 0)
cv2.setTrackbarPos("Cr_max", "Slider Window", 255)
cv2.setTrackbarPos("Cb_th", "Slider Window", 0)
cv2.setTrackbarPos("Cb_max", "Slider Window", 255)

# 슬라이더 작동
# q를 누르면 모든 윈도우 창 종료
while cv2.waitKey(1) != ord('q'):

    # 슬라이더 값 적용
    Y_th = cv2.getTrackbarPos("Y_th", "Slider Window")
    Y_max = cv2.getTrackbarPos("Y_max", "Slider Window")
    Cr_th = cv2.getTrackbarPos("Cr_th", "Slider Window")
    Cr_max = cv2.getTrackbarPos("Cr_max", "Slider Window")
    Cb_th = cv2.getTrackbarPos("Cb_th", "Slider Window")
    Cb_max = cv2.getTrackbarPos("Cb_max", "Slider Window")

    # # 적용된 슬라이더 값으로 하한, 상한 값 Tuple 지정
    # low_lower = (0, Cr_th, Cb_th)
    # low_upper = (Y_th, Cr_max, Cb_max)
    # high_lower = (Y_max, Cr_th, Cb_th)
    # high_upper = (179, Cr_max, Cb_max)
    lower = (Y_th, Cr_th, Cb_th)
    upper = (Y_max, Cr_max, Cb_max)

    # # 하한, 상한 범위 내의 픽셀을 Image Mask로 지정
    # img_mask_lower = cv2.inRange(img_hsv, low_lower, low_upper)
    # img_mask_higher = cv2.inRange(img_hsv, high_lower, high_upper)
    # img_mask = cv2.addWeighted(img_mask_lower, 1.0, img_mask_higher, 1.0, 0.0)

    img_mask = cv2.inRange(img_ycrcb, lower, upper)

    # 비트연산을 위해 이미지 배열 크기 변경
    img_merge = cv2.merge((img_mask, img_mask, img_mask))

    # 원본 얼굴 사진과 마스크에 해당하는 부분 AND 연산 -> 피부 이미지
    img_skin = cv2.bitwise_and(img, img_merge)
    
    # for iter in range(8):
    #     (b, g, r) = img_skin[eyebrow[iter][0], eyebrow[iter][1]]
    #     print("Pixel at ({}, {}) - Red: {}, Green: {}, Blue: {}".format(eyebrow[iter][0], eyebrow[iter][1], r, g, b))

    # 얼굴 영역을 ROI로 지정
    img_roi = img_skin[img_top:img_bottom, img_left:img_right]

    # 원본 얼굴 이미지 출력
    cv2.imshow("Original", img_mask)

    # 피부 이미지 출력
    cv2.imshow("Skin", img_roi)

cv2.destroyAllWindows()