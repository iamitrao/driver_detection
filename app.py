import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
import numpy as np
import time
import joblib
from math import sqrt
from collections import deque
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==============================
# LANDMARK INDICES & CONSTANTS
# ==============================
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]
NOSE_TIP = 1

HEAD_PITCH_DOWN_THRESH = -15
GAZE_Y_DOWN_THRESH = 6

UNSAFE_TIME_THRESHOLD = 3.0
RISK_WARNING = 0.45
RISK_ALERT = 0.55
RISK_WINDOW = 30
SIDE_DIST_TIME_THRESHOLD = 4.0

HIGH_RISK_THRESHOLD = 0.65
HIGH_RISK_TIME_THRESHOLD = 4.0

# ==============================
# UTILS
# ==============================
def euclidean(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_head_pose(image, face):
    h, w, _ = image.shape
    image_points = np.array([
        (face[1].x * w, face[1].y * h),
        (face[152].x * w, face[152].y * h),
        (face[33].x * w, face[33].y * h),
        (face[263].x * w, face[263].y * h),
        (face[61].x * w, face[61].y * h),
        (face[291].x * w, face[291].y * h)
    ], dtype="double")

    model_points = np.array([
        (0, 0, 0),
        (0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    _, rvec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles[0]*360, angles[1]*360, angles[2]*360

# ==============================
# VIDEO TRANSFORMER CLASS
# ==============================
class DriverMonitorTransformer(VideoTransformerBase):
    def __init__(self):
        try:
            self.model = joblib.load("driver_model.pkl")
        except:
            self.model = None
            
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1
        )
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1
        )
        
        self.risk_buffer = deque(maxlen=RISK_WINDOW)
        self.unsafe_start_time = None
        self.side_start_time = None
        self.high_risk_start_time = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        display = img.copy()
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)

        status = "NO FACE"
        color = (0, 255, 255)
        smoothed_risk = 0.0

        if face_results.multi_face_landmarks and self.model is not None:
            face = face_results.multi_face_landmarks[0].landmark
            def pt(i): return np.array([face[i].x * w, face[i].y * h])

            xs = [lm.x for lm in face]
            ys = [lm.y for lm in face]
            x_min, x_max = int(min(xs)*w), int(max(xs)*w)
            y_min, y_max = int(min(ys)*h), int(max(ys)*h)

            left_eye = [pt(i) for i in LEFT_EYE]
            right_eye = [pt(i) for i in RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            eye_state = 0 if ear < 0.18 else 1 if ear < 0.23 else 2

            gaze_y_val = (
                np.mean([pt(i)[1] for i in LEFT_IRIS + RIGHT_IRIS]) -
                np.mean([pt(i)[1] for i in LEFT_EYE + RIGHT_EYE])
            )

            pitch, yaw, roll = get_head_pose(img, face)

            rule_distracted = False
            if abs(yaw) < 20 and pitch < HEAD_PITCH_DOWN_THRESH and gaze_y_val > GAZE_Y_DOWN_THRESH:
                rule_distracted = True

            side_distracted = abs(yaw) > 30 and abs(pitch) < 10

            hand_face_dist = 0
            if hand_results.multi_hand_landmarks:
                hand = hand_results.multi_hand_landmarks[0].landmark
                hand_face_dist = euclidean(
                    np.array([hand[8].x*w, hand[8].y*h]),
                    pt(NOSE_TIP)
                )

            features = np.array([[ear, eye_state, 0, gaze_y_val, pitch, yaw, roll, hand_face_dist]])
            risk = self.model.predict_proba(features)[0][1]

            self.risk_buffer.append(risk)
            smoothed_risk = np.mean(self.risk_buffer)

            current_time = time.time()
            status = "SAFE"
            color = (0, 255, 0)

            if smoothed_risk >= HIGH_RISK_THRESHOLD:
                if self.high_risk_start_time is None:
                    self.high_risk_start_time = current_time

                if (current_time - self.high_risk_start_time) >= HIGH_RISK_TIME_THRESHOLD:
                    status = "DISTRACTED (HIGH RISK)"
                    color = (0, 0, 255)
            else:
                self.high_risk_start_time = None

            if status == "SAFE":
                if rule_distracted and smoothed_risk >= RISK_ALERT:
                    if self.unsafe_start_time is None:
                        self.unsafe_start_time = current_time

                    if (current_time - self.unsafe_start_time) >= UNSAFE_TIME_THRESHOLD:
                        status = "DISTRACTED"
                        color = (0, 0, 255)

                elif smoothed_risk >= RISK_WARNING:
                    status = "WARNING"
                    color = (0, 255, 255)
                    self.unsafe_start_time = None
                else:
                    self.unsafe_start_time = None

            if status == "SAFE" and side_distracted:
                if self.side_start_time is None:
                    self.side_start_time = current_time

                if (current_time - self.side_start_time) >= SIDE_DIST_TIME_THRESHOLD:
                    status = "DISTRACTED (SIDE)"
                    color = (0, 165, 255)
            else:
                self.side_start_time = None

            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), color, 4)
            
            font_scale = 1.0 if status == "SAFE" else 1.2
            thickness = 2 if status == "SAFE" else 3
            
            cv2.putText(display, status, (x_min, y_min - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(display, f"Risk: {smoothed_risk:.2f}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        elif self.model is None:
            cv2.putText(display, "Model not loaded", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        if "DISTRACTED" in status:
            cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 10)

        return display

def main():
    st.set_page_config(page_title="Driver AI Monitor", page_icon="🚗", layout="centered")

    st.title("🚗 Driver Distraction Monitor")
    st.markdown("""
    This application monitors driver distraction in real-time. It processes the webcam feed 
    directly in your browser and measures Eye Aspect Ratio (EAR), head pose tracking, and AI-predicted risk.
    
    *Please allow camera access and click 'START' to begin detection.*
    """)

    webrtc_streamer(
        key="driver-monitor",
        video_transformer_factory=DriverMonitorTransformer,
        async_transform=True,
    )

    st.markdown("---")
    st.markdown("### Metrics Explained")
    st.markdown("- **SAFE**: Normal driving behavior.")
    st.markdown("- **WARNING**: Elevated risk levels detected.")
    st.markdown("- **DISTRACTED**: Unsafe driving conditions identified. When distracted, the image frame will flash red.")

if __name__ == "__main__":
    main()
