import cv2, dlib, sys
import numpy as np
import time

scaler = 0.2
prev_time = 0
FPS = 10

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/python Project/opencv (2)/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
overlay = cv2.imread('D:/python Project/opencv (2)/mask/wmask.png',cv2.IMREAD_UNCHANGED)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    try :
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        return bg_img
    except Exception : return background_img

while True:
    #ret = 프레임 읽기을 성공하면 True 값 반환
    #img = 배열 형식의 영상 프레임 (가로 X 세로 X 3) 값 반환
    ret, img = cap.read()
    current_time = time.time() - prev_time
    if (ret is True) and (current_time > 1./ FPS) :
        prev_time = time.time()

        ori = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray)
        # 인식된 얼굴 개수 출력
        print("Number of faces detected: {}".format(len(faces)))
        for face in faces:
            dlib_shape = predictor(img_gray,face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

            top_left = np.min(shape_2d, axis=0)
            bottom_right = np.max(shape_2d, axis=0)

            face_size = max(bottom_right-top_left)
            center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

            # 슬라이싱
            result = overlay_transparent(ori, overlay, center_x, center_y+25, overlay_size=(face_size, face_size))

            img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()),color=(255,255,255),
            thickness=2,lineType=cv2.LINE_AA)

            for s in shape_2d:
                cv2.circle(img, center=tuple(s), radius=1, color=(255,255,255),thickness=2, lineType=cv2.LINE_AA)

            cv2.circle(img, center=tuple((center_x,center_y)),radius=1,color=(0,0,255),thickness=2,lineType=cv2.LINE_AA)
            cv2.imshow('result',result)
        cv2.imshow('img',img)

    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
