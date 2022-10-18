import cv2, dlib, math, matplotlib
import numpy as np

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
img = cv2.imread('./image/6.jpg')
dets = detector(img, 1)

for k, d in enumerate(dets): 
    shape = predictor(img, d)

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

    print('Left eye : ', left_eye)
    print('Right eye : ', right_eye)
    print('Mouth : ', mouth)

    left_eye_ellipse = cv2.fitEllipse(left_eye)
    right_eye_ellipse = cv2.fitEllipse(right_eye)
    mouth_ellipse = cv2.fitEllipse(mouth)

    cv2.ellipse(img, left_eye_ellipse, (0,0,0), -1)
    cv2.ellipse(img, right_eye_ellipse, (0,0,0), -1)
    cv2.ellipse(img, mouth_ellipse, (0,0,0), -1)

    img_roi = img[d.top():d.bottom(), d.left():d.right()]

    cv2.imshow('img_roi', img_roi)

cv2.namedWindow('img')
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()