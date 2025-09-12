# pose_backend.py
import os
import numpy as np
import cv2

from config import (
    POSE_TASK_PATH, MEDIAPIPE_USE_GPU, MEDIAPIPE_GL_BACKEND
)

# 환경 세팅
os.environ.setdefault("MEDIAPIPE_GL_BACKEND", MEDIAPIPE_GL_BACKEND)
if MEDIAPIPE_USE_GPU:
    os.environ.setdefault("MEDIAPIPE_USE_GPU", "1")

HAS_TASK_FILE = os.path.isfile(POSE_TASK_PATH)

# Mediapipe imports (Tasks는 실패 가능, solutions는 항상 가능)
import mediapipe as mp
from mediapipe import solutions as mp_solutions
try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    HAS_TASKS = True
except Exception:
    HAS_TASKS = False

class PoseBackend:
    """
    통합 백엔드:
      - 가능하면 Tasks(pose_landmarker_full.task) 사용
      - 실패/모델없음이면 solutions.pose(CPU)로 안전 폴백
    .process(frame_bgr) -> dict(keys: L, R, N) with (x,y,vis) or None
    """
    def __init__(self):
        self.mode = None
        self.ctx  = None
        # 1) Tasks 시도
        if HAS_TASKS and HAS_TASK_FILE:
            try:
                base_opts = mp_tasks.BaseOptions(model_asset_path=POSE_TASK_PATH)
                running_mode = mp_tasks.vision.RunningMode.VIDEO  # timestamp 필요
                opts = mp_vision.PoseLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=running_mode,
                    num_poses=1,
                )
                self.ctx = mp_vision.PoseLandmarker.create_from_options(opts)
                self.mode = "tasks"
                print(f"[pose] Tasks backend 사용: {POSE_TASK_PATH}")
            except Exception as e:
                print(f"[pose] GPU Tasks 실패 → solutions.pose 폴백: {e}")

        # 2) solutions.pose
        if self.ctx is None:
            self.ctx = mp_solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mode = "solutions"
            print("[pose] solutions.pose (CPU) 사용")

        self._ts_ms = 0  # tasks용 누적 timestamp

    def close(self):
        try:
            if self.mode == "solutions" and self.ctx:
                self.ctx.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    @staticmethod
    def _pick(lms, W, H):
        # mediapipe 인덱스
        PoseLandmark = mp.solutions.pose.PoseLandmark
        ids = {
            "L": PoseLandmark.LEFT_SHOULDER,
            "R": PoseLandmark.RIGHT_SHOULDER,
            "N": PoseLandmark.NOSE
        }
        out = {}
        for k, idx in ids.items():
            pt = lms[idx]
            x = float(pt.x * W); y = float(pt.y * H)
            vis = float(getattr(pt, "visibility", 1.0))
            out[k] = (x, y, vis)
        return out

    def process(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        if self.mode == "tasks":
            # Tasks API는 RGB + timestamp(ms)
            self._ts_ms += 33.3  # ~30fps 가정(실제값은 상관없고 증가만 하면 됨)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.ctx.detect_for_video(mp_img, int(self._ts_ms))
            if not res.pose_landmarks:
                return None
            lms = res.pose_landmarks[0]
            return self._pick(lms, W, H)

        # solutions
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.ctx.process(rgb)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks.landmark
        return self._pick(lms, W, H)

def make_pose_landmarker(use_gpu=True):
    # env는 파일 상단에서 이미 세팅됨
    bk = PoseBackend()
    return bk.mode, bk

def extract_displacements(video_path, pose_backend, pose_handle):
    import cv2, numpy as np, math
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps>0 else 30.0
    ts, dW, dY, dD = [], [], [], []
    t = 0.0; dt = 1.0/float(fps)

    def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

    while True:
        ok, frame = cap.read()
        if not ok: break
        out = pose_handle.process(frame)
        if out is not None and all(k in out for k in ("L","R","N")):
            L, R, N = out["L"], out["R"], out["N"]
            # 좌표: (x,y,vis) — y는 화면 아래가 +이므로 이후 bandpass로 DC 제거/정규화됨
            W = dist(L, R)                           # 어깨 간 거리(폭) → dW
            Y = N[1]                                 # 코의 수직 좌표 → dY
            mid = ((L[0]+R[0])/2.0, (L[1]+R[1])/2.0)
            D = dist(N, mid)                         # 코-어깨중점 거리 → dD
            ts.append(t); dW.append(W); dY.append(Y); dD.append(D)
        t += dt
    cap.release()

    if len(ts) < 3: return None, None, None, None
    return np.array(ts, np.float32), np.array(dW, np.float32), np.array(dY, np.float32), np.array(dD, np.float32)