from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox, get_foot_position
import numpy as np

class PlayerTracker:
    def __init__(self,model_path, use_polygon=True):
        self.model = YOLO(model_path)
        self.use_polygon = use_polygon

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # Tìm frame đầu tiên trong 10 frame đầu có đủ 2 cầu thủ
        chosen_player = {}
        for i in range(min(10, len(player_detections))):
            if self.use_polygon:
                chosen_player = self.choose_players(court_keypoints, player_detections[i])
            else:
                chosen_player = self.choose_players_by_model(player_detections[i])
            if len(chosen_player) == 2:
                print(f"✓ Chọn cầu thủ từ frame {i} | polygon={self.use_polygon}")
                break

        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {}
            for player_id, track_id in chosen_player.items():
                if track_id in player_dict:
                    filtered_player_dict[player_id] = player_dict[track_id]
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players_by_model(self, player_dict):
        """
        Dùng khi model đã detect đúng player, không cần polygon.
        Chọn 2 người xa nhau nhất theo foot position.
        """
        print(f"\n=== [Model mode] Phân tích {len(player_dict)} người được detect ===")

        if len(player_dict) == 0:
            return {}
        if len(player_dict) == 1:
            tid = list(player_dict.keys())[0]
            return {1: tid}

        track_ids = list(player_dict.keys())

        # Nếu đúng 2 người → dùng luôn
        if len(track_ids) == 2:
            sorted_by_y = sorted(track_ids, key=lambda tid: get_foot_position(player_dict[tid])[1])
            return {1: sorted_by_y[0], 2: sorted_by_y[1]}

        # Nếu hơn 2 → chọn 2 xa nhau nhất
        max_distance = 0
        chosen_pair = [track_ids[0], track_ids[1]]
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id1, id2 = track_ids[i], track_ids[j]
                dist = measure_distance(
                    get_foot_position(player_dict[id1]),
                    get_foot_position(player_dict[id2])
                )
                if dist > max_distance:
                    max_distance = dist
                    chosen_pair = [id1, id2]

        sorted_by_y = sorted(chosen_pair, key=lambda tid: get_foot_position(player_dict[tid])[1])
        print(f"✓ Player 1 (nửa trên): Track ID {sorted_by_y[0]}")
        print(f"✓ Player 2 (nửa dưới): Track ID {sorted_by_y[1]}")
        return {1: sorted_by_y[0], 2: sorted_by_y[1]}


    def is_point_in_court(self, point, court_keypoints,
                          padding_top=0.15, padding_bottom=0.2, padding_sides=0.05):
        """
        Kiểm tra foot position có nằm trong sân không, dùng padding KHÔNG đồng đều:
          - padding_top    = 0.25 : mở rộng nhiều phía trên vì Player 1 hay đứng sát/vượt baseline trên
          - padding_bottom = 0.05 : mở rộng ít phía dưới
          - padding_sides  = 0.05 : mở rộng ít 2 bên
        => Ball boy đứng SAU baseline sẽ bị loại vì chân họ nằm ngoài polygon.

        Keypoint layout (index trong mảng court_keypoints):
          [0,1]  = góc trên-trái
          [2,3]  = góc trên-phải
          [4,5]  = góc dưới-phải
          [6,7]  = góc dưới-trái
        """
        top_left     = np.array([court_keypoints[0], court_keypoints[1]], dtype=np.float32)
        top_right    = np.array([court_keypoints[2], court_keypoints[3]], dtype=np.float32)
        bottom_right = np.array([court_keypoints[6], court_keypoints[7]], dtype=np.float32)
        bottom_left  = np.array([court_keypoints[4], court_keypoints[5]], dtype=np.float32)

        # Kích thước sân (pixel) để đổi % → pixel thực tế
        court_width  = np.linalg.norm(top_right - top_left)
        court_height = np.linalg.norm(bottom_left - top_left)

        pad_top    = court_height * padding_top
        pad_bottom = court_height * padding_bottom
        pad_sides  = court_width  * padding_sides

        # Vector hướng chuẩn hoá (theo góc nghiêng của sân trong ảnh)
        down_vec  = (bottom_left - top_left) / np.linalg.norm(bottom_left - top_left)
        right_vec = (top_right   - top_left) / np.linalg.norm(top_right   - top_left)

        # Dịch chuyển từng góc đúng hướng padding của nó
        exp_top_left     = top_left     - down_vec * pad_top    - right_vec * pad_sides
        exp_top_right    = top_right    - down_vec * pad_top    + right_vec * pad_sides
        exp_bottom_right = bottom_right + down_vec * pad_bottom + right_vec * pad_sides
        exp_bottom_left  = bottom_left  + down_vec * pad_bottom - right_vec * pad_sides

        court_polygon = np.array([
            exp_top_left, exp_top_right,
            exp_bottom_right, exp_bottom_left
        ], dtype=np.int32)

        result = cv2.pointPolygonTest(court_polygon, point, False)
        return result >= 0

    def choose_players(self, court_keypoints, player_dict):
        """
        Chọn 2 cầu thủ có FOOT POSITION nằm trong polygon sân.
        Dùng foot position (điểm chân) thay vì center of bbox vì:
          - Chân cầu thủ luôn chạm đất trong sân
          - Ball boy đứng sau baseline → chân ngoài polygon → bị loại đúng
        """
        print(f"\n=== Phân tích {len(player_dict)} người được detect ===")

        valid_players = []
        for track_id, bbox in player_dict.items():
            foot_pos = get_foot_position(bbox)  # (x, y2) - điểm chân

            is_inside = self.is_point_in_court(foot_pos, court_keypoints)

            if is_inside:
                print(f"✓ Track ID {track_id} TRONG SÂN - foot: {foot_pos}")
                valid_players.append(track_id)
            else:
                print(f"✗ Track ID {track_id} NGOÀI SÂN - foot: {foot_pos}")

        valid_players.sort()

        # Không tìm thấy ai
        if len(valid_players) == 0:
            print("⚠️ Không tìm thấy cầu thủ nào trong sân!")
            return {}

        # Chỉ 1 người
        if len(valid_players) == 1:
            print(f"⚠️ Chỉ tìm thấy 1 cầu thủ: Track ID {valid_players[0]}")
            return {1: valid_players[0]}

        # Hơn 2 người → chọn 2 người xa nhau nhất (tính theo foot position)
        if len(valid_players) > 2:
            print(f"⚠️ Có {len(valid_players)} người trong sân, chọn 2 xa nhau nhất...")
            max_distance = 0
            chosen_pair = [valid_players[0], valid_players[1]]
            for i in range(len(valid_players)):
                for j in range(i + 1, len(valid_players)):
                    id1, id2 = valid_players[i], valid_players[j]
                    foot1 = get_foot_position(player_dict[id1])
                    foot2 = get_foot_position(player_dict[id2])
                    dist = measure_distance(foot1, foot2)
                    if dist > max_distance:
                        max_distance = dist
                        chosen_pair = [id1, id2]
            valid_players = chosen_pair

        # Sắp xếp theo Y của foot (nhỏ = nửa trên màn hình = Player 1)
        sorted_by_y = sorted(
            valid_players,
            key=lambda tid: get_foot_position(player_dict[tid])[1]
        )
        print(f"✓ Player 1 (nửa trên): Track ID {sorted_by_y[0]}")
        print(f"✓ Player 2 (nửa dưới): Track ID {sorted_by_y[1]}")
        return {1: sorted_by_y[0], 2: sorted_by_y[1]}

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
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

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is None:  # bỏ qua nếu chưa có track ID
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                    player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames