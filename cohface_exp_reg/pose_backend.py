# -*- coding: utf-8 -*-
"""
Mediapipe Pose → dW, dY, dD_perp 추출
- Tasks .task 가 있으면 GPU, 없으면 solutions CPU 폴백
"""
import cv2
import numpy as np
import os

from .config import MP_TASK_PATH

try:
    import mediapipe as mp
except Exception:
    mp = None

LM = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
}

def _landmark_xy(lm, W, H):
    return np.array([lm.x*W, lm.y*H], dtype=np.float32)

def _shoulder_axis_slow(SL_hist, SR_hist, fc=0.05, fs=30.0):
    if len(SL_hist) < 3: return 0.0
    v = (SR_hist - SL_hist)
    ang = np.arctan2(v[:,1], v[:,0])
    ang = np.unwrap(ang)
    k = max(1, int(round(fs/(2*np.pi*fc))))
    if k>1:
        ang_f = np.convolve(ang, np.ones(k)/k, mode='same')
    else:
        ang_f = ang
    return ang_f[-1]

def extract_displacements(video_path: str):
    if mp is None:
        raise RuntimeError("mediapipe is not installed.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    use_tasks = (os.path.exists(MP_TASK_PATH) and os.getenv("MEDIAPIPE_USE_GPU","1")=="1")

    if use_tasks:
        base = mp.tasks
        VisionRunningMode = base.vision.RunningMode
        pose_opts = base.vision.PoseLandmarkerOptions(
            base_options=base.BaseOptions(model_asset_path=MP_TASK_PATH),
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=False)
        detector = base.vision.PoseLandmarker.create_from_options(pose_opts)
    else:
        solutions = mp.solutions
        detector = solutions.pose.Pose()

    ts, dW, dY, dD, dD_perp = [], [], [], [], []
    SL_hist, SR_hist = [], []

    # 실제 FPS 사용, 실패 시 30Hz 폴백
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps <= 1e-3:
        fps = 30.0
    t = 0.0; dt = 1.0/float(fps)

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        if use_tasks:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det = detector.detect_for_video(mp_image, int(t*1000))
            if not det.pose_landmarks:
                t += dt; continue
            lms = det.pose_landmarks[0]
            def pick(i):
                pt = lms[i]
                return np.array([pt.x*W, pt.y*H], dtype=np.float32)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = detector.process(rgb)
            if not det.pose_landmarks:
                t += dt; continue
            lms = det.pose_landmarks.landmark
            def pick(i):
                pt = lms[i]
                return np.array([pt.x*W, pt.y*H], dtype=np.float32)

        nose = pick(LM["NOSE"])
        SL   = pick(LM["LEFT_SHOULDER"])
        SR   = pick(LM["RIGHT_SHOULDER"])

        mid = 0.5*(SL+SR)
        width = np.linalg.norm(SR-SL)
        dW.append(width)
        dY.append(mid[1])

        v = SR - SL
        n = np.array([-v[1], v[0]], dtype=np.float32)
        n = n / (np.linalg.norm(n)+1e-6)
        d = np.dot((nose - mid), n)
        dD.append(d)

        SL_hist.append(SL); SR_hist.append(SR)
        ang_slow = _shoulder_axis_slow(np.array(SL_hist), np.array(SR_hist), fc=0.05, fs=fps)
        v_slow = np.array([-np.sin(ang_slow), np.cos(ang_slow)], dtype=np.float32)
        d_perp = np.dot((nose - mid), v_slow)
        dD_perp.append(d_perp)

        ts.append(t)
        t += dt

    cap.release()
    return np.array(ts), np.array(dW), np.array(dY), np.array(dD), np.array(dD_perp)