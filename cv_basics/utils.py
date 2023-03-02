import time
import cv2
import mediapipe as mp
import open3d as o3d 
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from cv_basics.constants import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def preprocess(image: np.ndarray, intrinsic_mat: np.ndarray, distortion_coeff: np.ndarray, hands):
    image = cv2.undistort(image, intrinsic_mat, distortion_coeff)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def draw_hand(results, image):
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        print(f'HAND NUMBER: {hand_no+1}')
        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


def linear_regression_update_with_moving_average(X, y, prev, decay=ALPHA):
    reg = LinearRegression().fit(X, y)
    prev.coef_ = decay * reg.coef_ + (1 - decay) * prev.coef_
    prev.intercept_ = decay * reg.intercept_ + (1 - decay) * prev.intercept_


def hand_landmarks_to_array(hand_landmarks):
    ret = np.zeros((21, 3))
    for i in range(21): # 0~20
        ret[i][0] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x
        ret[i][1] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y
        ret[i][2] = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z
    return ret


def map_hand_to_2d(hands, img_width: int, img_heigth: int):
    ret = []
    if type(hands.multi_hand_landmarks) == type(None):
        return np.array([], dtype=np.float64)
    for hand_no, hand_landmarks in enumerate(hands.multi_hand_landmarks):
        ret.append(np.zeros((21, 2), dtype=np.float64))
        for i in range(21): # 0~20
            x = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * img_width
            y = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * img_heigth
            z = hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z
            ret[-1][i][0] = x
            ret[-1][i][1] = y
    return ret

# 이거 일단 양쪽 카메라에서 손이 1개씩만 있는 경우를 가정하고 해야겠다
# TODO disortion 보정한 intrinsic matrix 적용
def recon_3d_hand_points(img_w:int, img_h: int,
                    hands_l, hands_r,
                    l_int: np.ndarray=L_INTRINSIC, r_int:np.ndarray=R_INTRINSIC,
                    l_ext: np.ndarray=L_EXTRINSIC, r_ext: np.ndarray=R_EXTRINSIC):

    hand_l_xy = map_hand_to_2d(hands_l, img_w, img_h)
    hand_r_xy = map_hand_to_2d(hands_r, img_w, img_h)

    if not (len(hand_l_xy) == 1 and len(hand_r_xy) == 1):
        return None
    ret = cv2.triangulatePoints(l_int @ l_ext, r_int @ r_ext, hand_l_xy[0].T, hand_r_xy[0].T)

    # homogenous(x, y, z, k) -> (x/k, y/k, z/k)
    ret = ret[:3,:] / ret[3,:]
    return ret

def connect_3d_hand_points(points: np.ndarray):
    if type(points) == type(None):
        return None
    if points.shape != (3, 21):
        raise ValueError("array size must be 3 x 21. Try using transpose matrix if the size is 21 x 3.")
    points = points.T

    # Y좌표에 -1을 곱해줌
    # 2d 사진에서는 아래쪽이 +y 방향인데 3d 공간에서는 보통 위쪽이 +y방향이라 이렇게 함
    # open3d 시각화 결과 등에 영향을 줌
    points[:,1] *= -1
    points_o3d = o3d.utility.Vector3dVector(points)
    pointcloud = o3d.geometry.PointCloud(points=points_o3d)

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(list(mp_hands.HAND_CONNECTIONS))
    
    return pointcloud, lines

def recon_3d_hand(img_w: int, img_h: int,
                    hands_l, hands_r,
                    l_int: np.ndarray=L_INTRINSIC, r_int:np.ndarray=R_INTRINSIC,
                    l_ext: np.ndarray=L_EXTRINSIC, r_ext: np.ndarray=R_EXTRINSIC):
    points = recon_3d_hand_points(img_w, img_h, hands_l, hands_r, l_int, r_int, l_ext, r_ext)
    return connect_3d_hand_points(points)

