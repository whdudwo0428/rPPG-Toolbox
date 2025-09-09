# COHFACE: GT time viewer + 7-point motion segments (correlation to respiration GT)
# - 위(좌): 비디오 + 7포인트 + 상관 Top-3 선분(빨/노/파)
# - 우(상): Top-3 ΔL(t) 미니 그래프
# - 아래: pulse(t), respiration(t), HR_gt(t), RR_gt(t)  (모두 tt 축)
# 단축키:
#   [ ] / { }  → offset ±0.1s / ±1.0s       (영상↔GT 정합)
#   1/2/4/8     → HR/RR 창(12/2/4/8s)        (사전계산 재수행)
#   g           → 그래프 스팬 30/60/120s
#   a           → y-스케일 자동/고정 토글
#   m           → 모션 추적 ON/OFF
#   v           → ΔL 정의: 세로만 ↔ L2거리
#   r           → 7포인트 재배치(가슴 자동 초기화)
#   c           → ROI를 마우스로 드래그해서 지정(그 ROI 내부에서 7포인트 자동선정)
#   ESC         → 종료

import cv2, h5py, numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks
from collections import deque

# ========= 경로 =========
VIDEO = "/mnt/hdd18t/rppg_dataset/raw/cohface/1/0/data.mkv"
H5    = "/mnt/hdd18t/rppg_dataset/raw/cohface/1/0/data.hdf5"

# ========= 파라미터 =========
NUM_PTS      = 7
HR_BAND      = (0.70, 3.00)     # Hz (42–180 bpm)
RR_BAND      = (0.10, 0.50)     # Hz (6–30 brpm)
WIN_SEC      = 8                # HR/RR/RR_mtn 계산 창
SPAN_CHOICES = [30, 60, 120]    # 그래프 스팬
SPAN_SEC     = 60
CORR_WIN     = 20               # 상관 계산 창(초)
BG           = (16,16,16)

# ========= 유틸 =========
def put(img, txt, xy, s=0.78, col=(230,230,230), th=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)

def bandpass(x, fs, lo, hi, order=3):
    ny = fs*0.5
    b,a = butter(order, [lo/ny, hi/ny], btype="band")
    return filtfilt(b,a,x,method="gust")

def hr_from_win(x, fs):
    if len(x) < int(1.5*fs): return np.nan
    y = (x - x.mean())/(x.std()+1e-8)
    try: y = bandpass(y, fs, HR_BAND[0], HR_BAND[1], 3)
    except: pass
    pk,_ = find_peaks(y, distance=int(fs*0.33), prominence=np.std(y)*0.3)
    if pk.size < 2: return np.nan
    ibi = np.diff(pk)/fs
    return float(np.median(60.0/np.clip(ibi,1e-3,None)))

def rr_from_win(x, fs):
    if len(x) < int(1.5*fs): return np.nan
    nper = min(len(x), int(8*fs))
    f,P = welch(x - np.mean(x), fs=fs, nperseg=nper, noverlap=nper//2)
    m = (f>=RR_BAND[0])&(f<=RR_BAND[1])
    if not np.any(m): return np.nan
    fr,Pr = f[m], P[m]
    return 60.0*float(fr[np.argmax(Pr)])

def draw_axis(panel, ymin, ymax, title):
    h,w = panel.shape[:2]
    cv2.rectangle(panel,(0,0),(w-1,h-1),(40,40,40),1)
    put(panel, f"{ymin:.0f}", (6,h-8), 0.6, (180,180,180))
    put(panel, f"{ymax:.0f}", (6,18),   0.6, (180,180,180))
    put(panel, title, (8, 20), 0.6, (200,200,200))

def plot_series_scaled(panel, t, y, t0, span, ymin, ymax, color=(220,220,220), margin=8):
    h,w = panel.shape[:2]
    t1 = t0 + span
    m = (t>=t0)&(t<=t1)&np.isfinite(y)
    if not np.any(m): return
    T = (t[m]-t0)/max(1e-9, span)
    Y = (y[m]-ymin)/max(1e-6,(ymax-ymin))
    xs = (margin + (w-2*margin)*T).astype(int)
    ys = (h-1 - margin - (h-2*margin)*np.clip(Y,0,1)).astype(int)
    pts = np.stack([xs,ys],1)
    if len(pts)>=2: cv2.polylines(panel,[pts],False,color,2,cv2.LINE_AA)

# ========= GT 로드 =========
with h5py.File(H5,"r") as f:
    pulse = np.asarray(f["pulse"][:], np.float32)
    resp  = np.asarray(f["respiration"][:], np.float32)
    tt    = np.asarray(f["time"][:], np.float32)

fs = float(1.0/np.median(np.diff(tt)))
T_START, T_END = float(tt[0]), float(tt[-1])

# 파형(표시용)
pulse_vis = (pulse - np.mean(pulse))/max(1e-8,np.std(pulse))
try: pulse_vis = bandpass(pulse_vis, fs, HR_BAND[0], HR_BAND[1], 3)
except: pass
resp_std = (resp - np.mean(resp))/max(1e-8,np.std(resp))
resp_rr  = bandpass(resp_std, fs, RR_BAND[0], RR_BAND[1], 3)  # 상관용

# ========= HR/RR 사전계산 (tt 축) =========
def sliding_map(sig, tt, fs, win_sec, func):
    n=len(tt); out=np.full(n,np.nan,np.float32)
    if int(win_sec*fs)<8: return out
    j0=0
    for i in range(n):
        t_ref=tt[i]; t_start=t_ref-win_sec
        while j0<i and tt[j0]<t_start: j0+=1
        out[i]=func(sig[j0:i+1], fs)
    return out

print("Precomputing HR/RR on tt...")
HR_t = sliding_map(pulse, tt, fs, WIN_SEC, hr_from_win)
RR_t = sliding_map(resp,  tt, fs, WIN_SEC, rr_from_win)
print("Done.")

# ========= 비디오/모션 =========
cap = cv2.VideoCapture(VIDEO); assert cap.isOpened(), "영상 열기 실패"
win = "COHFACE: GT + motion-correlation"
cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1500, 980)

lk_params      = dict(winSize=(21,21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
feature_params = dict(maxCorners=NUM_PTS, qualityLevel=0.01, minDistance=18, blockSize=7)

enable_motion     = True
use_vertical_only = True
t_offset          = 0.0
span_idx          = SPAN_CHOICES.index(SPAN_SEC) if SPAN_SEC in SPAN_CHOICES else 1
auto_scale        = True

prev_gray = None
pts_prev  = None        # (N,1,2)
pts_init  = None
need_reinit = True

# ROI 선택(마우스)
roi_manual = None
_ixy, _selecting = (0,0), False
def on_mouse(event,x,y,flags,param):
    global roi_manual, _ixy, _selecting
    if event==cv2.EVENT_LBUTTONDOWN:
        _ixy=(x,y); _selecting=True
    elif event==cv2.EVENT_MOUSEMOVE and _selecting:
        x0,y0=_ixy; x1,y1=x,y
        roi_manual=(min(x0,x1),min(y0,y1), abs(x1-x0),abs(y1-y0))
    elif event==cv2.EVENT_LBUTTONUP:
        _selecting=False
cv2.setMouseCallback(win, on_mouse)

# 포인트/쌍 타임시리즈
times_vid = deque()                    # 비디오 시간축 (t_ref)
pairs     = None                       # [(i,j), ...]
pair_init_len = None                   # 초기 길이
pair_series = {}                       # (i,j) -> deque of ΔL
pair_colors = [(60,180,240),(60,220,120),(240,220,60)]  # 파/초/노 (우측 패널은 흰색들)

def auto_points(frame):
    """가슴/어깨 범위에서 NUM_PTS점 자동 선정"""
    H,W = frame.shape[:2]
    if roi_manual is None or roi_manual[2]<10 or roi_manual[3]<10:
        x1,x2=int(W*0.25),int(W*0.75)
        y1,y2=int(H*0.55),int(H*0.88)
    else:
        x1,y1,w,h=roi_manual; x2,y2=x1+w,y1+h
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    if pts is None or len(pts)<NUM_PTS:
        xs = np.linspace(x1+20, x2-20, NUM_PTS).astype(int)
        y  = int((y1+y2)/2)
        base = np.array([[[x,y]] for x in xs], np.float32)
    else:
        pts = pts[:NUM_PTS].reshape(-1,2); pts[:,0]+=x1; pts[:,1]+=y1
        base = pts.reshape(-1,1,2).astype(np.float32)
    return base

def build_pairs(n):
    out=[]
    for i in range(n):
        for j in range(i+1,n):
            out.append((i,j))
    return out

def ensure_pair_series():
    global pairs, pair_init_len, pair_series
    n = len(pts_init.reshape(-1,2))
    pairs = build_pairs(n)
    pair_init_len = {}
    pair_series = {}
    P = pts_init.reshape(-1,2)
    for (i,j) in pairs:
        L0 = float(np.linalg.norm(P[i]-P[j]))
        pair_init_len[(i,j)] = max(1e-6, L0)
        pair_series[(i,j)] = deque()

def update_motion_series(P):
    """현재 포인트 P(N,2)로 각 쌍 ΔL을 업데이트; 추적 실패는 np.nan 기록"""
    for (i,j) in pairs:
        if i>=len(P) or j>=len(P):
            pair_series[(i,j)].append(np.nan);
            continue
        L = float(np.linalg.norm(P[i]-P[j]))
        dL = L - pair_init_len[(i,j)]
        pair_series[(i,j)].append(dL)

def trim_series(max_age):
    """times_vid 및 각 pair deque를 오래된 것 제거"""
    while len(times_vid) and (times_vid[-1]-times_vid[0] > max_age):
        times_vid.popleft()
        for k in pair_series.keys():
            pair_series[k].popleft()

def corr_against_resp(pair_key, t_end, win_sec):
    """최근 win_sec 창에서 ΔL vs respiration(RR대역) 상관 (피어슨).
       times_vid 길이와 pair_series 길이가 달라도 안전하게 정렬."""
    if pair_key not in pair_series:
        return np.nan

    tv = np.asarray(list(times_vid), np.float32)
    yv = np.asarray(list(pair_series[pair_key]), np.float32)
    if tv.size == 0 or yv.size == 0:
        return np.nan

    # 길이 맞추기(뒤에서 정렬)
    L = min(len(tv), len(yv))
    tv = tv[-L:]
    yv = yv[-L:]

    t0 = t_end - win_sec
    m = (tv >= t0) & np.isfinite(yv)
    if np.count_nonzero(m) < 8:
        return np.nan

    tv = tv[m]
    yv = yv[m]

    # ΔL 전처리 → RR대역 밴드패스
    dt = np.median(np.diff(tv)) if tv.size > 1 else 0.0
    if dt <= 0:
        return np.nan
    fs_m = 1.0 / dt

    yv = yv - np.median(yv)
    try:
        yvf = bandpass(yv, fs_m, RR_BAND[0], RR_BAND[1], 3)
    except Exception:
        yvf = yv

    # Resp 파형(RR대역)을 동일 시간축으로 보간
    resp_seg = np.interp(tv, tt, resp_rr)

    if np.std(yvf) < 1e-6 or np.std(resp_seg) < 1e-6:
        return np.nan

    return float(np.corrcoef(yvf, resp_seg)[0, 1])


# ========= 루프 =========
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
prev_tvid = 0.0
span_idx = SPAN_CHOICES.index(SPAN_SEC) if SPAN_SEC in SPAN_CHOICES else 1

while True:
    ok, frame = cap.read()
    if not ok: break

    # 비디오 시간 & 기준시각
    tvid = (cap.get(cv2.CAP_PROP_POS_MSEC) or (prev_tvid + 1000.0/fps))/1000.0
    if tvid < prev_tvid: tvid = prev_tvid + 1.0/max(1.0,fps)
    prev_tvid = tvid
    t_ref = tvid + t_offset

    vis = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----- 모션 업데이트 -----
    if enable_motion:
        if need_reinit or prev_gray is None or pts_prev is None or len(pts_prev) < 3:
            pts_prev = auto_points(frame)
            pts_init = pts_prev.copy()
            ensure_pair_series()
            times_vid.clear()
            need_reinit = False

        if prev_gray is not None and pts_prev is not None:
            pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None,
                                                         winSize=(21,21), maxLevel=3,
                                                         criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
            good = (st.reshape(-1)==1) if st is not None else np.zeros((len(pts_prev),),bool)
            if np.count_nonzero(good) < 3:
                need_reinit = True
            else:
                P  = pts_next.reshape(-1,2)[good]
                P0 = pts_init.reshape(-1,2)[good]
                # 필요하면 init도 good만 남김
                pts_prev = pts_next[good].reshape(-1,1,2).astype(np.float32)
                pts_init = pts_init[good].reshape(-1,1,2).astype(np.float32)
                # ΔL 업데이트
                times_vid.append(float(t_ref))
                # base 길이 테이블도 good index 기준으로 다시 맞춤 필요
                # → pair_init_len/pairs 재계산
                ensure_pair_series() if len(P0)!=len(pair_init_len) else None
                # init 길이 다시 산출 (점 유지 중에만)
                Pfull = pts_prev.reshape(-1,2)
                if len(Pfull)>=2:
                    for (i,j) in pairs:
                        if i<len(Pfull) and j<len(Pfull):
                            L0 = float(np.linalg.norm(pts_init.reshape(-1,2)[i]-pts_init.reshape(-1,2)[j]))
                            pair_init_len[(i,j)] = max(1e-6, L0)
                # 실제 ΔL 기록
                update_motion_series(P)
                # 오래된 샘플 제거
                trim_series(max(SPAN_CHOICES)*2)

        # 포인트/ROI 표시
        if pts_prev is not None:
            for p in pts_prev.reshape(-1,2):
                cv2.circle(vis, (int(p[0]),int(p[1])), 3, (240,240,240), -1, cv2.LINE_AA)
        H,W = vis.shape[:2]
        if roi_manual is None or roi_manual[2]<10 or roi_manual[3]<10:
            x1,x2=int(W*0.25),int(W*0.75); y1,y2=int(H*0.55),int(H*0.88)
        else:
            x1,y1,w,h=roi_manual; x2,y2=x1+w,y1+h
        cv2.rectangle(vis,(x1,y1),(x2,y2),(160,160,160),1,cv2.LINE_AA)

    prev_gray = gray

    # ----- 상관 Top-3 계산 -----
    top_pairs=[]
    if enable_motion and len(times_vid)>=8 and pairs:
        t_end = times_vid[-1]
        scores=[]
        for k in pairs:
            c = corr_against_resp(k, t_end, CORR_WIN)
            if np.isfinite(c): scores.append((abs(c), c, k))   # 정렬은 abs, 보고는 부호 유지
        scores.sort(reverse=True)
        top_pairs = scores[:3]

    # ----- 레이아웃: 좌 비디오 / 우 Top3 ΔL / 하단 GT -----
    H,W = vis.shape[:2]
    side_w = 360
    PANEL_TEXT_H = 60
    TRACK_H = 100
    SEP = 8
    bottom_h = PANEL_TEXT_H + 4*TRACK_H + 3*SEP

    # 우측 패널 (Top3 ΔL)
    side = np.full((H, side_w, 3), BG, np.uint8)
    put(side, "Top-3 segments (dL)", (12, 26), 0.78)
    # 각 트랙(3개)을 위에서부터 그리기
    def draw_dl(track_idx, pair_key, color=(220,220,220)):
        h0 = 40 + track_idx*(H-60)//3
        h  = (H-100)//3
        panel = side[h0:h0+h, :]
        t0 = (times_vid[-1]-SPAN_CHOICES[span_idx]) if len(times_vid) else 0.0
        draw_axis(panel, -1, 1, f"{pair_key}  (corr {scores_map[pair_key]:+.2f})")
        if len(times_vid):
            tv = np.array(times_vid, np.float32)
            yv = np.array(pair_series[pair_key], np.float32)
            # 최근 스팬만, 정규화
            m = (tv >= tv[-1]-SPAN_CHOICES[span_idx]) & np.isfinite(yv)
            if np.any(m):
                ym = yv[m]; ym = ym - np.median(ym)
                denom = np.percentile(np.abs(ym), 95) + 1e-6
                ym = np.clip(ym/denom, -1, 1)
                plot_series_scaled(panel, tv[m], ym, tv[-1]-SPAN_CHOICES[span_idx], SPAN_CHOICES[span_idx], -1, 1, color=color)

    scores_map = {}
    for rank, (a, c, k) in enumerate(top_pairs):
        scores_map[k] = c
        col = [(60,220,240), (60,220,120), (240,220,60)][rank]
        draw_dl(rank, k, color=col)

    # Top-3 선분을 영상에 오버레이
    if top_pairs and pts_prev is not None:
        P = pts_prev.reshape(-1,2)
        for rank, (_, c, (i,j)) in enumerate(top_pairs):
            col = pair_colors[rank]
            if i<len(P) and j<len(P):
                cv2.line(vis, (int(P[i,0]),int(P[i,1])), (int(P[j,0]),int(P[j,1])), col, 3, cv2.LINE_AA)
                put(vis, f"{(i,j)}:{c:+.2f}", (int((P[i,0]+P[j,0])/2), int((P[i,1]+P[j,1])/2)-6), 0.6, col, 2)

    # ----- 하단 GT 패널 -----
    panel = np.full((bottom_h, W+side_w, 3), BG, np.uint8)
    # 텍스트
    # tt 기준 현재 HR/RR
    t_clip = np.clip(t_ref, T_START, T_END)
    i_ref = int(np.clip(np.searchsorted(tt, t_clip, side="left"), 0, len(tt)-1))
    hr_now = float(HR_t[i_ref]) if np.isfinite(HR_t[i_ref]) else float("nan")
    rr_now = float(RR_t[i_ref]) if np.isfinite(RR_t[i_ref]) else float("nan")
    put(panel, f"t_video:{tvid:06.2f}s   t_ref(tt):{t_clip:06.2f}s   offset:{t_offset:+.2f}s   "
               f"win:{WIN_SEC}s   span:{SPAN_CHOICES[span_idx]}s   motion:{'ON' if enable_motion else 'OFF'} "
               f"axis:{'V' if use_vertical_only else 'L2'}",
        (12, 26), 0.78)
    put(panel, f"HR_gt:{hr_now:.1f} bpm    RR_gt:{rr_now:.1f} brpm", (12, 52), 0.78)

    # y-스케일
    t1 = t_clip; t0 = max(T_START, t1 - SPAN_CHOICES[span_idx])
    if auto_scale:
        mhr = (tt>=t0)&(tt<=t1)&np.isfinite(HR_t); HR_MIN,HR_MAX=(float(np.nanmin(HR_t[mhr])-5), float(np.nanmax(HR_t[mhr])+5)) if np.any(mhr) else (40,140)
        mrr = (tt>=t0)&(tt<=t1)&np.isfinite(RR_t); RR_MIN,RR_MAX=(float(np.nanmin(RR_t[mrr])-2), float(np.nanmax(RR_t[mrr])+2)) if np.any(mrr) else (6,30)
    else:
        HR_MIN,HR_MAX=(40,140); RR_MIN,RR_MAX=(6,30)

    # 4트랙
    y = 60; TH=100; SEP=8
    # 1) pulse
    p1 = panel[y:y+TH, :W]; y += TH+SEP
    draw_axis(p1, -2, 2, "pulse (norm, bandpassed) [tt]")
    plot_series_scaled(p1, tt, pulse_vis, t0, SPAN_CHOICES[span_idx], -2, 2)
    # 2) respiration
    p2 = panel[y:y+TH, :W]; y += TH+SEP
    draw_axis(p2, -2, 2, "respiration (norm) [tt]")
    plot_series_scaled(p2, tt, resp_std, t0, SPAN_CHOICES[span_idx], -2, 2)
    # 3) HR
    p3 = panel[y:y+TH, :W]; y += TH+SEP
    draw_axis(p3, HR_MIN, HR_MAX, "HR_gt (bpm) [tt]")
    plot_series_scaled(p3, tt, HR_t, t0, SPAN_CHOICES[span_idx], HR_MIN, HR_MAX)
    # 4) RR
    p4 = panel[y:y+TH, :W]; y += TH+SEP
    draw_axis(p4, RR_MIN, RR_MAX, "RR_gt (brpm) [tt]")
    plot_series_scaled(p4, tt, RR_t, t0, SPAN_CHOICES[span_idx], RR_MIN, RR_MAX)

    # 커서
    def draw_cursor(p, t0, span):
        h,w=p.shape[:2]; mrg=8
        x=int(mrg + (w-2*mrg)*((t_clip - t0)/max(1e-9, span)))
        cv2.line(p,(x,0),(x,h-1),(180,180,80),1,cv2.LINE_AA)
    for sp in (p1,p2,p3,p4): draw_cursor(sp, t0, SPAN_CHOICES[span_idx])

    # ----- 합성 및 표시 -----
    top_row = np.hstack([vis, side])
    bottom   = panel
    canvas   = np.vstack([top_row, bottom])
    cv2.imshow(win, canvas)

    # ----- 입력 -----
    key = cv2.waitKey(1) & 0xFF
    if key==27: break
    elif key in (ord('1'),ord('2'),ord('4'),ord('8')):
        WIN_SEC = {ord('1'):12, ord('2'):2, ord('4'):4, ord('8'):8}[key]
        HR_t = sliding_map(pulse, tt, fs, WIN_SEC, hr_from_win)
        RR_t = sliding_map(resp,  tt, fs, WIN_SEC, rr_from_win)
    elif key==ord('g'):
        span_idx = (span_idx + 1) % len(SPAN_CHOICES)
    elif key==ord('a'):
        auto_scale = not auto_scale
    elif key==ord('m'):
        enable_motion = not enable_motion
    elif key==ord('v'):
        use_vertical_only = not use_vertical_only
    elif key==ord('r'):
        need_reinit = True; times_vid.clear()
        pair_series.clear()
    elif key==ord('c'):
        # 드래그로 ROI 잡기 → r로 재배치
        pass
    elif key==ord('['):  t_offset -= 0.1
    elif key==ord(']'):  t_offset += 0.1
    elif key==ord('{'):  t_offset -= 1.0
    elif key==ord('}'):  t_offset += 1.0

cap.release()
cv2.destroyAllWindows()
