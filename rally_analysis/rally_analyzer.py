# rally_analysis/rally_analyzer.py
"""
Phân tích 1 rally tennis từ dữ liệu tracking + mini-court.
Tính các chỉ số chiến thuật/thể lực và sinh insights tự động cho HLV.
"""
import cv2
import numpy as np
import sys
sys.path.append('../')
import constants


class RallyAnalyzer:
    """Tính chỉ số per-rally từ player/ball tracking + speed data."""

    # Threshold (có thể tune sau)
    SPRINT_SPEED_KMH   = 20.0    # = 4 m/s
    SPRINT_MIN_FRAMES  = 5       # số frame liên tục tối thiểu để tính 1 sprint
    DEFENSE_RATIO      = 1.5     # ratio > đây = phòng thủ
    BASELINE_ATTACK_M  = 0.5     # sát baseline trong khoảng này = tấn công
    BASELINE_DEFENSE_M = 1.5     # lùi quá đây = phòng thủ

    def __init__(self, fps, mini_court):
        self.fps = fps
        self.mini_court = mini_court
        self.pixel_per_meter = (
            mini_court.get_width_of_mini_court() / constants.DOUBLE_LINE_WIDTH
        )

    def analyze(self, player_mini, ball_mini, speeds, total_distances):
        """
        Tính toàn bộ chỉ số cho 1 rally.

        Returns: dict chứa các metric đã tính.
        """
        n_frames = len(player_mini)
        duration = n_frames / self.fps if self.fps > 0 else 0

        # Peak & avg speed cho từng player (lọc speed > 0)
        peak_speeds, avg_speeds = {}, {}
        for pid in [1, 2]:
            vals = [s[pid] for s in speeds if pid in s and s[pid] > 0]
            if vals:
                peak_speeds[pid] = round(float(np.max(vals)), 1)
                avg_speeds[pid]  = round(float(np.mean(vals)), 1)
            else:
                peak_speeds[pid] = 0.0
                avg_speeds[pid]  = 0.0

        # Sprint count: số đoạn liên tục >= 3 frame có speed > 14.4 km/h
        sprint_counts = {
            pid: self._count_sprints(speeds, pid) for pid in [1, 2]
        }

        # Movement ratio: tỉ số quãng đường (lớn / nhỏ, luôn >= 1)
        d1 = max(total_distances.get(1, 0), 0.01)  # tránh chia 0
        d2 = max(total_distances.get(2, 0), 0.01)
        if d1 >= d2:
            movement_ratio = d1 / d2
            high_runner = 1
        else:
            movement_ratio = d2 / d1
            high_runner = 2

        # Vị trí trung bình + phân loại tấn công/phòng thủ
        avg_positions   = {}
        position_styles = {}
        for pid in [1, 2]:
            pos = self._get_avg_position(player_mini, pid)
            avg_positions[pid] = pos
            position_styles[pid] = (
                self._classify_position(pid, pos) if pos else "không xác định"
            )

        # Court coverage (m²) — diện tích convex hull
        coverage_areas = {
            pid: self._compute_coverage_m2(player_mini, pid) for pid in [1, 2]
        }

        # Ball end zone — 1 trong 6 vùng
        ball_end_zone = self._get_ball_end_zone(ball_mini)

        return {
            'duration_seconds': round(duration, 1),
            'n_frames':         n_frames,
            'distances':        total_distances,
            'peak_speeds':      peak_speeds,
            'avg_speeds':       avg_speeds,
            'sprint_counts':    sprint_counts,
            'movement_ratio':   round(movement_ratio, 2),
            'high_runner':      high_runner,
            'avg_positions':    avg_positions,
            'position_styles':  position_styles,
            'coverage_areas':   coverage_areas,
            'ball_end_zone':    ball_end_zone,
        }

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _count_sprints(self, speeds, pid):
        """Đếm số đoạn sprint (>= SPRINT_MIN_FRAMES frame liên tục > threshold)."""
        sprints, run = 0, 0
        for fs in speeds:
            if pid in fs and fs[pid] > self.SPRINT_SPEED_KMH:
                run += 1
            else:
                if run >= self.SPRINT_MIN_FRAMES:
                    sprints += 1
                run = 0
        if run >= self.SPRINT_MIN_FRAMES:
            sprints += 1
        return sprints

    def _get_avg_position(self, player_mini, pid):
        positions = [f[pid] for f in player_mini if pid in f]
        if not positions:
            return None
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _classify_position(self, pid, avg_pos):
        """
        Player tấn công (sát baseline) hay phòng thủ (lùi sâu)?
        P1 baseline = court_start_y (top), P2 baseline = court_end_y (bottom).
        """
        if avg_pos is None:
            return "không xác định"
        x, y = avg_pos
        ppm = self.pixel_per_meter

        if pid == 1:
            baseline_y = self.mini_court.court_start_y
            offset_m = (baseline_y - y) / ppm   # dương = lùi ra ngoài baseline
        else:
            baseline_y = self.mini_court.court_end_y
            offset_m = (y - baseline_y) / ppm

        if offset_m > self.BASELINE_DEFENSE_M:
            return "phòng thủ"
        elif offset_m < self.BASELINE_ATTACK_M:
            return "tấn công"
        else:
            return "trung lập"

    def _compute_coverage_m2(self, player_mini, pid):
        """Diện tích convex hull của vùng player hoạt động (m²)."""
        positions = [f[pid] for f in player_mini if pid in f]
        if len(positions) < 3:
            return 0.0
        points = np.array(positions, dtype=np.float32)
        hull = cv2.convexHull(points)
        area_px = cv2.contourArea(hull)
        return round(float(area_px / (self.pixel_per_meter ** 2)), 1)

    def _get_ball_end_zone(self, ball_mini):
        """Vị trí ball cuối → 1 trong 6 vùng (2x3)."""
        ball_pos = None
        for frame in reversed(ball_mini):
            if frame and 1 in frame:
                ball_pos = frame[1]
                break
        if ball_pos is None:
            return None

        x, y = ball_pos
        mc = self.mini_court

        center_x = (mc.court_start_x + mc.court_end_x) / 2
        h_label = "trái" if x < center_x else "phải"

        court_h = mc.court_end_y - mc.court_start_y
        third = court_h / 3
        if y < mc.court_start_y + third:
            v_label = "phía P1"
        elif y < mc.court_start_y + 2 * third:
            v_label = "giữa sân"
        else:
            v_label = "phía P2"

        return f"{v_label} - {h_label}"


# ═════════════════════════════════════════════════════════════════════
# Insight generator (rule-based)
# ═════════════════════════════════════════════════════════════════════

def generate_insights(stats):
    """
    Sinh các câu nhận xét từ stats.
    Trả về list string đã sort theo priority (cao → thấp).
    """
    if not stats:
        return []

    items = []  # (priority, message) — priority thấp = quan trọng hơn

    # 1. Movement Ratio — HIGH
    ratio = stats['movement_ratio']
    runner = stats['high_runner']
    if ratio > 2.0:
        items.append((1, f"Player {runner} bị ép rất mạnh — chạy gấp {ratio:.1f}x đối thủ"))
    elif ratio > 1.5:
        items.append((2, f"Player {runner} ở thế phòng thủ — chạy gấp {ratio:.1f}x đối thủ"))
    elif ratio < 1.2:
        items.append((4, "Cả 2 cầu thủ chạy tương đương — rally cân bằng"))

    # 2. Position Style — HIGH
    for pid in [1, 2]:
        style = stats['position_styles'].get(pid)
        if style == 'phòng thủ':
            items.append((2, f"Player {pid} lùi sâu sau baseline → ở thế phòng thủ"))
        elif style == 'tấn công':
            items.append((3, f"Player {pid} đứng sát baseline → ở thế tấn công"))

    # 3. Sprint — MEDIUM
    for pid in [1, 2]:
        n = stats['sprint_counts'].get(pid, 0)
        if n >= 3:
            items.append((3, f"Player {pid} có {n} pha sprint mạnh — rally cường độ cao"))
        elif n >= 1:
            items.append((5, f"Player {pid} có {n} pha bứt tốc"))

    # 4. Peak speed — MEDIUM
    peaks = stats['peak_speeds']
    max_peak = max(peaks.values()) if peaks else 0
    if max_peak >= 18:
        fast_pid = max(peaks, key=peaks.get)
        items.append((4, f"Peak speed cao: Player {fast_pid} đạt {max_peak} km/h"))

    # 5. Coverage — MEDIUM
    cov = stats['coverage_areas']
    if cov:
        max_pid = max(cov, key=cov.get)
        if cov[max_pid] >= 25:
            items.append((5, f"Player {max_pid} chơi rộng — phủ {cov[max_pid]:.0f} m² mặt sân"))

    # 6. Ball end zone — MEDIUM
    zone = stats.get('ball_end_zone')
    if zone:
        items.append((4, f"Bóng kết thúc ở {zone}"))

    # 7. Duration — LOW
    dur = stats['duration_seconds']
    if dur > 20:
        items.append((6, f"Rally dài {dur:.0f}s — đòi hỏi thể lực tốt"))
    elif dur < 5:
        items.append((6, f"Rally ngắn {dur:.0f}s — pha bóng nhanh, có thể winner sớm"))

    items.sort(key=lambda x: x[0])
    return [msg for _, msg in items]