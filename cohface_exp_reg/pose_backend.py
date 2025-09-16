# -*- coding: utf-8 -*-
"""
Mediapipe Pose → dW, dY, dD_perp 추출
- Tasks .task 가 있으면 GPU, 없으면 solutions CPU 폴백
- Tasks 결과 객체 버전 차이를 getattr로 안전 처리 (pose_landmarks / landmarks)
"""
import os
from typing import Sequence

import cv2
import numpy as np

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


def _shoulder_axis_slow(SL_hist, SR_hist, fc=0.05, fs=30.0):
    if len(SL_hist) < 3:
        return 0.0
    v = (SR_hist - SL_hist)
    ang = np.arctan2(v[:, 1], v[:, 0])
    ang = np.unwrap(ang)
    # 간단 moving-average 저역통과
    k = max(1, int(round(fs / (2 * np.pi * fc))))
    if k > 1:
        ang_f = np.convolve(ang, np.ones(k) / k, mode="same")
    else:
        ang_f = ang
    return float(ang_f[-1])


def _pick_from_tasks_landmarks(lms_list: Sequence, idx: int, W: int, H: int) -> np.ndarray:
    """Tasks 결과: lms_list[0][idx]가 NormalizedLandmark로 가정."""
    lms0 = lms_list[0]
    pt = lms0[idx]
    return np.array([float(pt.x) * W, float(pt.y) * H], dtype=np.float32)


def _pick_from_solutions_landmarks(lms_container, idx: int, W: int, H: int) -> np.ndarray:
    """Solutions 결과: det.pose_landmarks.landmark[idx]"""
    pt = lms_container.landmark[idx]
    return np.array([float(pt.x) * W, float(pt.y) * H], dtype=np.float32)


def extract_displacements(video_path: str):
    if mp is None:
        raise RuntimeError("mediapipe is not installed.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    # Tasks(.task) + GPU 사용 여부
    use_tasks = os.path.exists(MP_TASK_PATH) and os.getenv("MEDIAPIPE_USE_GPU", "1") == "1"

    if use_tasks:
        # Tasks API
        base = mp.tasks
        VisionRunningMode = base.vision.RunningMode
        pose_opts = base.vision.PoseLandmarkerOptions(
            base_options=base.BaseOptions(model_asset_path=MP_TASK_PATH),
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        detector = base.vision.PoseLandmarker.create_from_options(pose_opts)
    else:
        # Solutions API
        solutions = mp.solutions
        detector = solutions.pose.Pose()

    ts, dW, dY, dD, dD_perp = [], [], [], [], []
    SL_hist, SR_hist = [], []

    # 실제 FPS 사용, 실패 시 30Hz 폴백
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps <= 1e-3:
        fps = 30.0
    t = 0.0
    dt = 1.0 / float(fps)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        if use_tasks:
            # Tasks 경로
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det = detector.detect_for_video(mp_image, int(t * 1000))

            # 버전-안전: pose_landmarks → landmarks 순으로 조회
            lms_list = getattr(det, "pose_landmarks", None)
            if lms_list is None:
                lms_list = getattr(det, "landmarks", None)

            if not lms_list or len(lms_list) == 0:
                t += dt
                continue

            def pick(i: int) -> np.ndarray:
                return _pick_from_tasks_landmarks(lms_list, i, W, H)

        else:
            # Solutions 경로
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = detector.process(rgb)

            lms_container = getattr(det, "pose_landmarks", None)
            if lms_container is None:
                t += dt
                continue

            def pick(i: int) -> np.ndarray:
                return _pick_from_solutions_landmarks(lms_container, i, W, H)

        # 선택 지점 추출
        nose = pick(LM["NOSE"])
        SL = pick(LM["LEFT_SHOULDER"])
        SR = pick(LM["RIGHT_SHOULDER"])

        mid = 0.5 * (SL + SR)
        width = float(np.linalg.norm(SR - SL))
        dW.append(width)
        dY.append(float(mid[1]))

        # 순간 수직축
        v = SR - SL
        n = np.array([-v[1], v[0]], dtype=np.float32)
        n = n / (np.linalg.norm(n) + 1e-6)
        d = float(np.dot((nose - mid), n))
        dD.append(d)

        # 느린 어깨축 기반 수직 성분(dD_perp)
        SL_hist.append(SL)
        SR_hist.append(SR)
        ang_slow = _shoulder_axis_slow(np.array(SL_hist), np.array(SR_hist), fc=0.05, fs=fps)
        v_slow = np.array([-np.sin(ang_slow), np.cos(ang_slow)], dtype=np.float32)
        d_perp = float(np.dot((nose - mid), v_slow))
        dD_perp.append(d_perp)

        ts.append(t)
        t += dt

    cap.release()
    return (
        np.asarray(ts, dtype=np.float32),
        np.asarray(dW, dtype=np.float32),
        np.asarray(dY, dtype=np.float32),
        np.asarray(dD, dtype=np.float32),
        np.asarray(dD_perp, dtype=np.float32),
    )
