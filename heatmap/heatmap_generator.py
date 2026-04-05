import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, court_width, court_height):
        """
        court_width, court_height: kích thước vùng mini court (pixel)
        """
        self.court_width = int(court_width)
        self.court_height = int(court_height)

    def generate_heatmap(self, player_mini_court_detections, player_id, mini_court,
                         pad_x=30, pad_y=40):
        """
        Tạo ảnh heatmap từ toàn bộ tọa độ của 1 cầu thủ trên mini court.
        Vùng density được mở rộng ra ngoài đường biên sân bằng pad_x, pad_y (pixel).

        pad_x: padding mở rộng sang 2 bên trái/phải (pixel trên mini court)
        pad_y: padding mở rộng lên/xuống baseline trên/dưới (pixel trên mini court)
        """
        # Dùng background rectangle làm ranh giới tối đa (đã rộng hơn court)
        # Sau đó thêm pad bổ sung nếu cần vượt qua padding_court=20px sẵn có
        base_x = mini_court.start_x
        base_y = mini_court.start_y
        base_end_x = mini_court.end_x
        base_end_y = mini_court.end_y

        # Mở rộng thêm pad ra ngoài background rectangle, nhưng clamp về 0
        region_x = max(0, base_x - pad_x)
        region_y = max(0, base_y - pad_y)
        region_end_x = base_end_x + pad_x
        region_end_y = base_end_y + pad_y

        w = region_end_x - region_x
        h = region_end_y - region_y

        # Ma trận tích lũy điểm xuất hiện
        density = np.zeros((h, w), dtype=np.float32)

        for frame_detections in player_mini_court_detections:
            if player_id not in frame_detections:
                continue
            x, y = frame_detections[player_id]
            # Chuyển sang tọa độ local trong vùng mở rộng
            lx = int(x - region_x)
            ly = int(y - region_y)
            if 0 <= lx < w and 0 <= ly < h:
                density[ly, lx] += 1

        # Làm mịn bằng Gaussian để tạo hiệu ứng "vùng nóng" lan rộng
        density = cv2.GaussianBlur(density, (51, 51), 0)

        # Normalize 0→255 rồi áp colormap
        if density.max() > 0:
            density = density / density.max()
        heatmap_img = (density * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

        # Lưu vùng region để overlay dùng lại
        self._last_region = (region_x, region_y, w, h)

        return heatmap_colored  # ảnh BGR, shape (h, w, 3)

    def overlay_heatmap_on_frame(self, frame, heatmap_colored, mini_court, alpha=0.5):
        """
        Overlay heatmap lên frame tại vùng region mở rộng.
        """
        if not hasattr(self, '_last_region'):
            # fallback về cách cũ nếu generate chưa được gọi
            region_x = mini_court.court_start_x
            region_y = mini_court.court_start_y
        else:
            region_x, region_y, w, h = self._last_region

        h_map, w_map = heatmap_colored.shape[:2]

        # Clamp để không vượt ra ngoài frame
        frame_h, frame_w = frame.shape[:2]
        end_y = min(region_y + h_map, frame_h)
        end_x = min(region_x + w_map, frame_w)
        actual_h = end_y - region_y
        actual_w = end_x - region_x

        roi = frame[region_y:end_y, region_x:end_x]
        heatmap_crop = heatmap_colored[:actual_h, :actual_w]

        # Chỉ blend vùng có màu (loại bỏ màu đen = không có dữ liệu)
        mask = cv2.cvtColor(heatmap_crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        blended = cv2.addWeighted(roi, 1 - alpha, heatmap_crop, alpha, 0)
        roi[mask > 0] = blended[mask > 0]
        frame[region_y:end_y, region_x:end_x] = roi

        return frame

    def draw_heatmap_on_last_frame(self, video_frames, player_mini_court_detections, mini_court,
                                   pad_x=30, pad_y=40):
        h, w = video_frames[0].shape[:2]
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 30

        canvas = mini_court.draw_background_rectangle(canvas)
        canvas = mini_court.draw_court(canvas)

        for player_id in [1, 2]:
            heatmap = self.generate_heatmap(
                player_mini_court_detections, player_id, mini_court,
                pad_x=pad_x, pad_y=pad_y
            )
            canvas = self.overlay_heatmap_on_frame(canvas, heatmap, mini_court, alpha=0.6)

        # Crop chỉ lấy vùng mini court + padding, KHÔNG resize
        x1 = max(0, mini_court.start_x - pad_x)
        y1 = max(0, mini_court.start_y - pad_y)
        x2 = min(w, mini_court.end_x + pad_x)
        y2 = min(h, mini_court.end_y + pad_y)
        cropped = canvas[y1:y2, x1:x2]

        return cropped