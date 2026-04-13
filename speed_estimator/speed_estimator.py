import cv2
import sys
sys.path.append('../')
import constants
from utils import measure_distance

class SpeedEstimator:
    def __init__(self, fps):
        self.fps = fps
        self.window = 5  # tính tốc độ dựa trên khoảng cách 5 frame

    def calculate_speed(self, player_mini_court_detections, court_drawing_width):
        """
        player_mini_court_detections: list[dict] — [{player_id: (x, y)}, ...]
        court_drawing_width: pixel width của mini court (để convert pixel → meter)
        """
        # pixel_per_meter trong mini court
        pixel_per_meter = court_drawing_width / constants.DOUBLE_LINE_WIDTH

        total_frames = len(player_mini_court_detections)
        speeds = [{} for _ in range(total_frames)]  # mỗi frame: {player_id: speed_km_h}

        for frame_num in range(self.window, total_frames):
            curr = player_mini_court_detections[frame_num]
            prev = player_mini_court_detections[frame_num - self.window]

            for player_id in curr:
                if player_id not in prev:
                    continue  # skip nếu frame trước không có player

                curr_pos = curr[player_id]
                prev_pos = prev[player_id]

                distance_px   = measure_distance(curr_pos, prev_pos)
                distance_m    = distance_px / pixel_per_meter
                time_s        = self.window / self.fps
                speed_km_h    = (distance_m / time_s) * 3.6

                # Gán tốc độ cho tất cả frame trong window (để hiển thị mượt)
                for i in range(frame_num - self.window, frame_num + 1):
                    speeds[i][player_id] = speed_km_h

        return speeds

    def draw_speed_on_frames(self, video_frames, speeds, player_detections):
        """
        Vẽ tốc độ gần bbox của player trên main frame
        """
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            if frame_num >= len(speeds):
                output_frames.append(frame)
                continue

            speed_dict  = speeds[frame_num]
            player_dict = player_detections[frame_num]

            for player_id, bbox in player_dict.items():
                if player_id not in speed_dict:
                    continue

                speed = speed_dict[player_id]
                x1, y1, x2, y2 = bbox

                # Vẽ tốc độ ngay dưới label "Player ID: x"
                text = f"{speed:.1f} km/h"
                cv2.putText(
                    frame, text,
                    (int(x1), int(y2) + 20),        # dưới bbox
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

            output_frames.append(frame)
        return output_frames
    
    def calculate_total_distance(self, player_mini_court_detections, court_drawing_width):
        """
        Tính tổng quãng đường mỗi cầu thủ đã chạy (mét).
        """
        pixel_per_meter = court_drawing_width / constants.DOUBLE_LINE_WIDTH
        total_distance = {1: 0.0, 2: 0.0}
        
        for frame_num in range(1, len(player_mini_court_detections)):
            curr = player_mini_court_detections[frame_num]
            prev = player_mini_court_detections[frame_num - 1]
            
            for player_id in curr:
                if player_id not in prev:
                    continue
                distance_px = measure_distance(curr[player_id], prev[player_id])
                distance_m = distance_px / pixel_per_meter
                if player_id in total_distance:
                    total_distance[player_id] += distance_m
        
        # Round kết quả
        return {pid: round(dist, 2) for pid, dist in total_distance.items()}