import cv2, dlib, math, matplotlib
import numpy as np

# 이미지 불러오기
img = cv2.imread('./image/19.jpg')

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
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(img_hsv)

# Otsu 알고리즘을 이용하여 적절한 하한, 상한 값 Tuple 지정
ret_S, mat_S = cv2.threshold(S, -1, 255, cv2.THRESH_OTSU)
ret_V, mat_V = cv2.threshold(V, -1, 255, cv2.THRESH_OTSU)
S_th = ret_S
V_th = ret_V
S_max = V_max = 255

# Otsu threshold value (디버그용)
print('S thres value : ', ret_S)
print('V thres value : ', ret_V)

# 가중치
S_val = 0.4
V_val = 1.3

# 적용된 값으로 하한, 상한 값 Tuple 지정
low_lower = (0, int(S_th*S_val), int(V_th*V_val))
low_upper = (30, 255, 255)
high_lower = (150, int(S_th*S_val), int(V_th*V_val))
high_upper = (179, 255, 255)

# 하한, 상한 범위 내의 픽셀을 Image Mask로 지정
img_mask_lower = cv2.inRange(img_hsv, low_lower, low_upper)
img_mask_higher = cv2.inRange(img_hsv, high_lower, high_upper)
img_mask = cv2.addWeighted(img_mask_lower, 1.0, img_mask_higher, 1.0, 0.0)

# 비트연산을 위해 이미지 배열 크기 변경
img_merge = cv2.merge((img_mask, img_mask, img_mask))

# 원본 얼굴 사진과 마스크에 해당하는 부분 AND 연산 -> 피부 이미지
img_skin = cv2.bitwise_and(img, img_merge)

# 얼굴 영역을 ROI로 지정
img_roi = img_skin[img_top:img_bottom, img_left:img_right]

# 피부 얼굴 색 평균 계산
r_sum = 0
g_sum = 0
b_sum = 0
pixel_count = 0

for y in range(img_bottom-img_top-1):
    for x in range(img_right-img_left-1):
        (b, g, r) = img_roi[x][y]
        if ~(b == 0 and g == 0 and r == 0):
            r_sum += r
            g_sum += g
            b_sum += b
            pixel_count += 1


RGB_avg = (r_sum/pixel_count, g_sum/pixel_count, b_sum/pixel_count)
print('Average RGB : ', RGB_avg)
RGB_Mat = np.full((1,1,3), (r_sum/pixel_count/255, g_sum/pixel_count/255, b_sum/pixel_count/255), np.float32) 
LAB = cv2.cvtColor(RGB_Mat, cv2.COLOR_RGB2LAB)
(l,a,b) = LAB[0][0]
print('LAB : ({}, {}, {})'.format(l,a,b))
if a > b :
    print('당신은 쿨톤입니다.')
else :
    print('당신은 웜톤입니다.')


# 원본 얼굴 이미지 출력
cv2.imshow("Original", img_mask)

# 피부 이미지 출력
cv2.imshow("Skin", img_roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
