from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox
import numpy as np

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def is_point_in_court(self, point, court_keypoints, padding_percent=0.4):
        """
        Kiểm tra xem điểm có nằm trong sân đấu không (có padding)
        Sử dụng 4 góc sân (keypoints 0,1,2,3) để tạo polygon
        padding_percent: mở rộng sân ra bao nhiêu % (mặc định 15%)
        """
        # Lấy 4 góc sân - GIẢ SỬ 4 điểm đầu là 4 góc
        # NẾU KEYPOINTS CỦA BẠN KHÁC, SỬA INDEX Ở ĐÂY
        corners = np.array([
            [court_keypoints[0], court_keypoints[1]],   # Góc 1
            [court_keypoints[2], court_keypoints[3]],   # Góc 2
            [court_keypoints[4], court_keypoints[5]],   # Góc 3
            [court_keypoints[6], court_keypoints[7]]    # Góc 4
        ], dtype=np.float32)
        
        # Tính tâm sân
        center = corners.mean(axis=0)
        
        # Mở rộng polygon ra từ tâm
        expanded_corners = []
        for corner in corners:
            direction = corner - center
            expanded_corner = center + direction * (1 + padding_percent)
            expanded_corners.append(expanded_corner)
        
        court_polygon = np.array(expanded_corners, dtype=np.int32)
        
        # Dùng cv2.pointPolygonTest để kiểm tra
        result = cv2.pointPolygonTest(court_polygon, point, False)
        return result >= 0  # >= 0 nghĩa là trong hoặc trên biên

    def choose_players(self, court_keypoints, player_dict):
        """
        Chọn 2 cầu thủ NẰM TRONG keypoints polygon
        """
        print(f"\n=== Phân tích {len(player_dict)} người được detect ===")
        
        valid_players = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            
            # Kiểm tra xem có trong sân không
            is_inside = self.is_point_in_court(player_center, court_keypoints)
            
            if is_inside:
                print(f"✓ Track ID {track_id} TRONG SÂN - center: {player_center}")
                valid_players.append(track_id)
            else:
                print(f"✗ Track ID {track_id} NGOÀI SÂN - center: {player_center}")
        
        # Sắp xếp để đảm bảo consistent order
        valid_players.sort()
        
        if len(valid_players) == 0:
            print("⚠️ KHÔNG TÌM THẤY CẦU THỦ NÀO TRONG SÂN!")
            print(f"Keypoints sân: {court_keypoints[:8]}")  # In 4 góc đầu
            return []
        
        if len(valid_players) == 1:
            print(f"⚠️ CHỈ TÌM THẤY 1 CẦU THỦ: {valid_players}")
            return valid_players
        
        if len(valid_players) == 2:
            print(f"✓✓ CHỌN 2 CẦU THỦ: {valid_players}")
            return valid_players
        
        # Nếu có > 2 người, chọn 2 người xa nhau nhất
        if len(valid_players) > 2:
            print(f"⚠️ CÓ {len(valid_players)} NGƯỜI TRONG SÂN, chọn 2 xa nhau nhất...")
            max_distance = 0
            chosen_pair = [valid_players[0], valid_players[1]]
            
            for i in range(len(valid_players)):
                for j in range(i+1, len(valid_players)):
                    id1, id2 = valid_players[i], valid_players[j]
                    center1 = get_center_of_bbox(player_dict[id1])
                    center2 = get_center_of_bbox(player_dict[id2])
                    dist = measure_distance(center1, center2)
                    
                    if dist > max_distance:
                        max_distance = dist
                        chosen_pair = [id1, id2]
            
            print(f"✓✓ CHỌN: {chosen_pair} (xa nhau {max_distance:.2f}px)")
            return chosen_pair
        
        return valid_players


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames