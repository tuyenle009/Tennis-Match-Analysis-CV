# trajectory/trajectory_map.py
import cv2
import numpy as np


class TrajectoryMapGenerator:
    """
    Tạo trajectory map: đường đi của 2 player + vị trí bóng start/end
    trên mini-court. Dùng để phân tích 1 rally tennis.

    Player path: polyline với màu sáng dần theo thời gian
                 (frame đầu mờ, frame cuối đậm).
    Marker:
        ○ start (vòng rỗng)
        ● end (chấm đặc)
        ★ ball start (vàng)
        ✕ ball end - nơi rally kết thúc (đỏ)
    """

    # Màu BGR (khớp UI: P1=🔵, P2=🔴)
    PLAYER_COLORS = {
        1: (255, 130, 50),    # Blue
        2: (60, 90, 240),     # Red
    }
    BALL_START_COLOR = (50, 220, 255)    # Vàng
    BALL_END_COLOR   = (50, 50, 255)     # Đỏ

    def __init__(self, line_thickness=3, marker_size=10, point_stride=3):
        self.line_thickness = line_thickness
        self.marker_size = marker_size
        self.point_stride = point_stride

    def generate(self, player_mini_court_detections,
                 ball_mini_court_detections,
                 mini_court, frame_shape,
                 pad_x=30, pad_y=40):
        """
        Returns:
            BGR image (numpy array) đã crop về vùng mini-court + padding.
        """
        h, w = frame_shape[:2]
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 30  # Nền tối

        # Vẽ background + lines của mini-court (dùng method có sẵn)
        canvas = mini_court.draw_background_rectangle(canvas)
        canvas = mini_court.draw_court(canvas)

        # Vẽ trajectory từng player
        for pid in [1, 2]:
            color = self.PLAYER_COLORS.get(pid, (200, 200, 200))
            points = self._extract_player_path(player_mini_court_detections, pid)
            if len(points) >= 2:
                canvas = self._draw_gradient_polyline(canvas, points, color)
                canvas = self._draw_start_marker(canvas, points[0], color)
                canvas = self._draw_end_marker(canvas, points[-1], color)

        # Marker bóng start/end
        # Marker bóng start: dùng vị trí player gần bóng nhất ở frame đầu
        # (proxy cho server position - tránh sai số homography do ball mid-air)
        serve_position = self._get_serve_position(
            player_mini_court_detections, ball_mini_court_detections
        )
        if serve_position is not None:
            self._draw_star(canvas, serve_position, size=12,
                            color=self.BALL_START_COLOR)

        # Marker bóng end: dùng vị trí ball cuối cùng (gần mặt đất, homography ít sai)
        ball_points = self._extract_ball_path(ball_mini_court_detections)
        if len(ball_points) >= 1:
            self._draw_x_marker(canvas, ball_points[-1])

        # Crop về vùng mini-court + padding
        x1 = max(0, mini_court.start_x - pad_x)
        y1 = max(0, mini_court.start_y - pad_y)
        x2 = min(w, mini_court.end_x + pad_x)
        y2 = min(h, mini_court.end_y + pad_y)
        cropped = canvas[y1:y2, x1:x2]

        cropped = self._add_legend(cropped)
        return cropped

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _extract_player_path(self, detections, pid):
        points = [f[pid] for f in detections if pid in f]
        # Subsample mỗi point_stride frame để polyline không bị rối
        # Vẫn giữ điểm cuối để marker end đặt đúng vị trí
        if self.point_stride > 1 and len(points) > self.point_stride * 2:
            sampled = points[::self.point_stride]
            if sampled[-1] != points[-1]:
                sampled.append(points[-1])
            return sampled
        return points

    def _extract_ball_path(self, ball_detections):
        return [f[1] for f in ball_detections if f and 1 in f]

    def _get_serve_position(self, player_detections, ball_detections):
        """
        Vị trí 'ball start' = vị trí player gần bóng nhất ở frame đầu
        detect được cả ball và player. Người đó là server (vừa phát bóng).

        Lý do: homography ánh xạ pixel → court ground plane. Bóng mid-air
        project ra sẽ lệch (camera nghiêng, sân hình thang). Vị trí player
        luôn ở mặt đất nên homography đúng → dùng làm proxy chính xác hơn
        cho vị trí xuất phát của bóng.
        """
        for i, ball_frame in enumerate(ball_detections):
            if not (ball_frame and 1 in ball_frame):
                continue
            if i >= len(player_detections):
                continue
            player_frame = player_detections[i]
            if not player_frame:
                continue

            ball_pos = ball_frame[1]
            closest_pid = min(
                player_frame.keys(),
                key=lambda pid: ((player_frame[pid][0] - ball_pos[0]) ** 2 +
                                (player_frame[pid][1] - ball_pos[1]) ** 2)
            )
            return player_frame[closest_pid]

        return None


    def _draw_gradient_polyline(self, canvas, points, base_color):
        """Polyline sáng dần: 30% → 100% brightness."""
        n_seg = len(points) - 1
        if n_seg < 1:
            return canvas
        for i in range(n_seg):
            t = i / max(1, n_seg - 1)
            brightness = 0.3 + 0.7 * t
            color = tuple(int(c * brightness) for c in base_color)
            p1 = (int(points[i][0]),     int(points[i][1]))
            p2 = (int(points[i + 1][0]), int(points[i + 1][1]))
            cv2.line(canvas, p1, p2, color, self.line_thickness, cv2.LINE_AA)
        return canvas

    def _draw_start_marker(self, canvas, point, color):
        """○ - vòng rỗng có chấm trắng giữa."""
        p = (int(point[0]), int(point[1]))
        cv2.circle(canvas, p, self.marker_size, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
        return canvas

    def _draw_end_marker(self, canvas, point, color):
        """● - chấm đặc viền trắng."""
        p = (int(point[0]), int(point[1]))
        cv2.circle(canvas, p, self.marker_size, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p, self.marker_size, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def _draw_star(self, canvas, center, size, color):
        """Ngôi sao 5 cánh."""
        cx, cy = int(center[0]), int(center[1])
        pts = []
        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5
            r = size if i % 2 == 0 else size * 0.4
            pts.append([int(cx + r * np.cos(angle)),
                        int(cy + r * np.sin(angle))])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_x_marker(self, canvas, point):
        """✕ với viền trắng."""
        p = (int(point[0]), int(point[1]))
        s = 10
        cv2.circle(canvas, p, s + 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (p[0]-s, p[1]-s), (p[0]+s, p[1]+s),
                 self.BALL_END_COLOR, 3, cv2.LINE_AA)
        cv2.line(canvas, (p[0]-s, p[1]+s), (p[0]+s, p[1]-s),
                 self.BALL_END_COLOR, 3, cv2.LINE_AA)

    def _add_legend(self, canvas):
        """Legend nhỏ ở góc dưới trái."""
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (5, h - 95), (185, h - 5), (15, 15, 25), -1)
        canvas[:] = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Player 1
        cv2.circle(canvas, (20, h - 78), 6, self.PLAYER_COLORS[1], -1)
        cv2.putText(canvas, "Player 1", (35, h - 74), font, 0.42,
                    (230, 230, 230), 1, cv2.LINE_AA)
        # Player 2
        cv2.circle(canvas, (20, h - 58), 6, self.PLAYER_COLORS[2], -1)
        cv2.putText(canvas, "Player 2", (35, h - 54), font, 0.42,
                    (230, 230, 230), 1, cv2.LINE_AA)
        # Ball start
        self._draw_star(canvas, (20, h - 36), 7, self.BALL_START_COLOR)
        cv2.putText(canvas, "Ball start", (35, h - 32), font, 0.42,
                    (230, 230, 230), 1, cv2.LINE_AA)
        # Ball end
        cv2.line(canvas, (14, h - 20), (26, h - 8),
                 self.BALL_END_COLOR, 2, cv2.LINE_AA)
        cv2.line(canvas, (14, h - 8), (26, h - 20),
                 self.BALL_END_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Ball end (out)", (35, h - 11), font, 0.42,
                    (230, 230, 230), 1, cv2.LINE_AA)
        return canvas