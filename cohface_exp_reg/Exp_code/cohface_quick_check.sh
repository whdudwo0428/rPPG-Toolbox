# ===== cohface_quick_check.sh =====
set -euo pipefail

ROOT="/mnt/hdd18t/rppg_dataset/raw/cohface"
LINK="~/PycharmProjects/rPPG-Toolbox/dataset/cohface"

echo "== [1] 심볼릭 링크 상태 =="
if [ -L $LINK ]; then
  echo "OK: $LINK -> $(readlink -f $LINK)"
else
  if [ -e $LINK ]; then
    echo "WARN: $LINK 는 심볼릭이 아니라 일반 폴더/파일입니다."
  else
    echo "MISS: $LINK 링크(또는 폴더)가 없습니다."
  fi
fi

echo
echo "== [2] 프로토콜 파일 유효성 =="
for P in all/train.txt all/dev.txt all/test.txt; do
  if [ -f "$ROOT/protocols/$P" ]; then
    CNT=$(grep -c '.' "$ROOT/protocols/$P" || true)
    echo "OK: protocols/$P  (lines=$CNT)"
  else
    echo "MISS: protocols/$P"
  fi
done

echo
echo "== [3] 샘플 3개 무작위 점검 (video/hdf5 페어) =="
LIST="$ROOT/protocols/all/train.txt"
if [ -f "$LIST" ]; then
  shuf -n 3 "$LIST" | while read -r rel; do
    rel="${rel%/}"                          # strip trailing slash
    base="$ROOT/${rel%/data}"
    vid="$base/data.mkv"
    [ -f "$vid" ] || vid="$base/data.avi"
    h5="$base/data.hdf5"
    echo "-- sample: $rel"
    [ -f "$vid" ] && echo "   video: OK ($vid)" || echo "   video: MISS ($vid)"
    [ -f "$h5" ]  && echo "   hdf5 : OK ($h5)"  || echo "   hdf5 : MISS ($h5)"
  done
else
  echo "MISS: $LIST"
fi

echo
echo "== [4] 코덱/프레임 점검 (ffprobe) =="
HAS_FFPROBE=$(command -v ffprobe || true)
if [ -z "$HAS_FFPROBE" ]; then
  echo "MISS: ffprobe가 없습니다. 설치 권장: sudo apt install -y ffmpeg"
else
  FIRST=$(head -n 1 "$ROOT/protocols/all/train.txt" | sed 's|/data$||')
  [ -n "$FIRST" ] || { echo "MISS: train.txt 비어있음"; exit 0; }
  base="$ROOT/$FIRST"
  vid="$base/data.mkv"; [ -f "$vid" ] || vid="$base/data.avi"
  echo "ffprobe on: $vid"
  ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,avg_frame_rate,r_frame_rate -of default=nw=1 "$vid" || true
fi

echo
echo "== [5] HDF5 키 점검 (pulse/respiration/time) =="
python3 - <<'PY'
import h5py, os, sys
ROOT="/mnt/hdd18t/rppg_dataset/raw/cohface"
lst=os.path.join(ROOT,'protocols','all','train.txt')
if not os.path.isfile(lst):
    print("MISS: train.txt not found"); sys.exit(0)
with open(lst) as f:
    line=f.readline().strip().rstrip('/')
if not line:
    print("MISS: train.txt is empty"); sys.exit(0)
base=os.path.join(ROOT, line[:-5])  # remove '/data'
h5=os.path.join(base, 'data.hdf5')
if not os.path.isfile(h5):
    print("MISS: hdf5 not found:", h5); sys.exit(0)
with h5py.File(h5,'r') as H:
    keys=list(H.keys())
print("HDF5 file:", h5)
print("keys:", keys)
need={'pulse','respiration','time'}
print("has_all_keys:", need.issubset(set(keys)))
PY
