"""
Script vẽ court polygon lên frame đầu tiên để debug vùng detect.
Chạy: python debug_court_polygon.py

Hiển thị:
  - Tất cả 14 keypoints với index (đánh số từ 0)
  - 4 keypoints góc sân được highlight VÀNG + label tên góc
    → giúp xác nhận index nào = top_left / top_right / bottom_left / bottom_right
  - Polygon gốc (ĐỎ) = 4 góc sân không padding
  - Polygon mở rộng (XANH LÁ) = vùng thực tế dùng để lọc cầu thủ
"""
import cv2
import numpy as np
import sys
sys.path.append('../')


# ════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH — chỉnh ở đây
# ════════════════════════════════════════════════════════════════════════════
VIDEO_PATH     = 'input_videos/inp_vid4.mp4'
OUTPUT_PATH    = 'output_videos/debug_polygon.jpg'
PADDING_TOP    = 0.15
PADDING_BOTTOM = 0.20
PADDING_SIDES  = 0.05
# ════════════════════════════════════════════════════════════════════════════


def draw_court_polygon(frame, court_keypoints,
                       padding_top=0.15, padding_bottom=0.2, padding_sides=0.05,
                       color=(0, 255, 0), thickness=2):
    """Vẽ polygon mở rộng (xanh lá) và polygon gốc (đỏ) lên frame."""

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

    # Polygon mở rộng — XANH LÁ
    polygon = np.array([
        exp_top_left, exp_top_right,
        exp_bottom_right, exp_bottom_left
    ], dtype=np.int32)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)

    # Polygon gốc — ĐỎ
    orig_polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    cv2.polylines(frame, [orig_polygon], isClosed=True, color=(0, 0, 255), thickness=1)

    return frame


def draw_corner_labels(frame, court_keypoints):
    """
    Highlight 4 keypoints góc sân bằng vòng tròn VÀNG + label:
      - Index trong mảng (ví dụ: idx=0)
      - Tên biến đang được gán trong code của player_tracker.py

    Nhìn vào ảnh output:
      Nếu label "bottom_LEFT (code)" nằm ở bên PHẢI thực tế
      → index bị swap → cần sửa lại trong player_tracker.py
    """
    corner_info = [
        (0, "idx=0 -> top_left"),
        (1, "idx=1 -> top_right"),
        (2, "idx=2 -> bottom_LEFT (code)"),
        (3, "idx=3 -> bottom_RIGHT (code)"),
    ]

    for kp_idx, label in corner_info:
        x = int(court_keypoints[kp_idx * 2])
        y = int(court_keypoints[kp_idx * 2 + 1])

        # Vòng tròn lớn màu VÀNG
        cv2.circle(frame, (x, y), 14, (0, 215, 255), 3)

        # Nền đen cho dễ đọc
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame,
                      (x + 18, y - th - 6),
                      (x + 18 + tw + 4, y + 4),
                      (0, 0, 0), -1)

        # Chữ vàng
        cv2.putText(frame, label, (x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

    return frame


def draw_legend(frame, padding_top, padding_bottom, padding_sides):
    """Vẽ legend góc trên trái."""
    lines = [
        ("YELLOW circle = 4 corner keypoints used in polygon", (0, 215, 255)),
        ("RED dot + number = all keypoints (index from 0)",    (0, 80, 255)),
        ("GREEN polygon = expanded zone (with padding)",        (0, 200, 0)),
        ("RED polygon   = original court boundary",             (0, 0, 200)),
        (f"Padding: top={padding_top}  bot={padding_bottom}  sides={padding_sides}", (200, 200, 200)),
    ]
    y0 = 35
    for text, color in lines:
        cv2.putText(frame, text, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)  # shadow
        cv2.putText(frame, text, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y0 += 30
    return frame


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
from court_line_detector import CourtLineDetector
from utils import read_video

print("Đọc video...")
frames = read_video(VIDEO_PATH)
frame  = frames[0].copy()

print("Detect court keypoints...")
detector  = CourtLineDetector('models/keypoints_model_04.pth')
court_kps = detector.predict(frame)

print("\nCác keypoints được detect:")
for i in range(14):
    print(f"  idx {i:2d}: ({court_kps[i*2]:.1f}, {court_kps[i*2+1]:.1f})")

# 1. Vẽ tất cả 14 keypoints (đỏ nhỏ) — index bắt đầu từ 0
for i in range(0, len(court_kps), 2):
    x, y = int(court_kps[i]), int(court_kps[i+1])
    cv2.circle(frame, (x, y), 5, (0, 80, 255), -1)
    cv2.putText(frame, str(i // 2), (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2)

# 2. Vẽ polygon gốc (đỏ) + mở rộng (xanh lá)
frame = draw_court_polygon(frame, court_kps,
                           padding_top=PADDING_TOP,
                           padding_bottom=PADDING_BOTTOM,
                           padding_sides=PADDING_SIDES,
                           color=(0, 200, 0), thickness=3)

# 3. Highlight 4 góc với label tên biến trong code
frame = draw_corner_labels(frame, court_kps)

# 4. Legend
frame = draw_legend(frame, PADDING_TOP, PADDING_BOTTOM, PADDING_SIDES)

cv2.imwrite(OUTPUT_PATH, frame)
print(f"\n✅ Đã lưu: {OUTPUT_PATH}")
print("\nCách đọc kết quả:")
print("  → Nhìn label 'idx=2 -> bottom_LEFT (code)' nằm ở góc nào thực tế?")
print("  → Nếu nó nằm bên PHẢI → index 2 và 3 đang bị SWAP trong player_tracker.py")
print("  → Nếu nó nằm bên TRÁI → index đúng, không cần sửa")