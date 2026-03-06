"""
Script vẽ court polygon lên frame đầu tiên để debug vùng detect.
Chạy: python debug_court_polygon.py
"""
import cv2
import numpy as np
import sys
sys.path.append('../')

def draw_court_polygon(frame, court_keypoints,
                       padding_top=0.25, padding_bottom=0.05, padding_sides=0.05,
                       color=(0, 255, 0), thickness=2):
    top_left     = np.array([court_keypoints[0], court_keypoints[1]], dtype=np.float32)
    top_right    = np.array([court_keypoints[2], court_keypoints[3]], dtype=np.float32)
    bottom_left  = np.array([court_keypoints[4], court_keypoints[5]], dtype=np.float32)
    bottom_right = np.array([court_keypoints[6], court_keypoints[7]], dtype=np.float32)

    court_width  = np.linalg.norm(top_right - top_left)
    court_height = np.linalg.norm(bottom_left - top_left)

    pad_top    = court_height * padding_top
    pad_bottom = court_height * padding_bottom
    pad_sides  = court_width  * padding_sides

    down_vec  = (bottom_left - top_left) / np.linalg.norm(bottom_left - top_left)
    right_vec = (top_right   - top_left) / np.linalg.norm(top_right   - top_left)

    exp_top_left     = top_left     - down_vec * pad_top    - right_vec * pad_sides
    exp_top_right    = top_right    - down_vec * pad_top    + right_vec * pad_sides
    exp_bottom_right = bottom_right + down_vec * pad_bottom + right_vec * pad_sides
    exp_bottom_left  = bottom_left  + down_vec * pad_bottom - right_vec * pad_sides

    polygon = np.array([
        exp_top_left, exp_top_right,
        exp_bottom_right, exp_bottom_left
    ], dtype=np.int32)

    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)

    # Vẽ polygon gốc (không padding) màu đỏ để so sánh
    orig_polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    cv2.polylines(frame, [orig_polygon], isClosed=True, color=(0, 0, 255), thickness=1)

    return frame

# ── THAY ĐỔI CÁC GIÁ TRỊ NÀY ──────────────────────────────────────────
VIDEO_PATH     = 'input_videos/inp_vid4.mp4'
OUTPUT_PATH    = 'output_videos/debug_polygon.jpg'
PADDING_TOP    = 0.15   # thử: 0.10, 0.15, 0.20, 0.25
PADDING_BOTTOM = 0.2
PADDING_SIDES  = 0.05
# ────────────────────────────────────────────────────────────────────────

from court_line_detector import CourtLineDetector
from utils import read_video

print("Đọc video...")
frames = read_video(VIDEO_PATH)
frame  = frames[0].copy()

print("Detect court keypoints...")
detector  = CourtLineDetector('models/keypoints_model_04.pth')
court_kps = detector.predict(frame)

# Vẽ keypoints gốc
for i in range(0, len(court_kps), 2):
    x, y = int(court_kps[i]), int(court_kps[i+1])
    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(frame, str(i//2 + 1), (x+8, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Vẽ polygon mở rộng (xanh lá)
frame = draw_court_polygon(frame, court_kps,
                           padding_top=PADDING_TOP,
                           padding_bottom=PADDING_BOTTOM,
                           padding_sides=PADDING_SIDES,
                           color=(0, 255, 0), thickness=3)

cv2.putText(frame, f"GREEN = padding top={PADDING_TOP} bot={PADDING_BOTTOM} sides={PADDING_SIDES}",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(frame, "RED = court boundary (no padding)",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imwrite(OUTPUT_PATH, frame)
print(f"✅ Đã lưu: {OUTPUT_PATH}")