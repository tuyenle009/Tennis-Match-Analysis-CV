from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker, ball_tracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from speed_estimator import SpeedEstimator
from heatmap import HeatmapGenerator
from utils import smooth_player_trajectory

import cv2

def main():
    number_of_vid = 7
    input_video_path = f'input_videos/inp_vid{number_of_vid}.mp4'
    video_frames = read_video(input_video_path)

    # ✅ THÊM: Đọc FPS động từ video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"✓ FPS: {fps}")

    # Detect players and balls in the video frames
    player_tracker = PlayerTracker(model_path='models/yolo26x.pt',use_polygon=True)
    ball_tracker = BallTracker(model_path='models/yolo26m_best_100e.pt')
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=False,
                                                     stub_path=f'tracker_stubs/player_detections_{number_of_vid}.pkl')

    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path=f"tracker_stubs/ball_detections_{number_of_vid}.pkl")

    # Court keypoints
    court_model_path = 'models/keypoints_model_04.pth'
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose and filter players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Minicourt
    mini_court = MiniCourt(video_frames[0])
    mini_court.set_homography(court_keypoints) 

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)


    # Sau dòng convert_bounding_boxes...
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    # THÊM:
    player_mini_court_detections = smooth_player_trajectory(player_mini_court_detections, window=5)
    print("✓ Trajectory smoothed")

    #: Tính tốc độ player
    speed_estimator = SpeedEstimator(fps=fps)
    speeds = speed_estimator.calculate_speed(
        player_mini_court_detections,
        mini_court.get_width_of_mini_court()
    )

    # Draw bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    # ✅ THÊM: Vẽ tốc độ lên frame (sau khi draw bbox, trước frame number)
    output_video_frames = speed_estimator.draw_speed_on_frames(
        output_video_frames, speeds, player_detections
    )

    # Draw frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save
    save_video(output_video_frames, f'output_videos/output_inp_vid{number_of_vid}.avi')


    # THAY BẰNG đoạn này:
    # heatmap_gen = HeatmapGenerator(
    #     court_width=mini_court.court_end_x - mini_court.court_start_x,
    #     court_height=mini_court.court_end_y - mini_court.court_start_y
    # )

    # frame_height = video_frames[0].shape[0]

    # heatmap_img = heatmap_gen.generate_heatmap(
    #     player_detections,    # tọa độ frame gốc, KHÔNG phải mini_court
    #     court_keypoints,
    #     frame_height,
    #     padding=60
    # )
    

    # cv2.imwrite(f'output_videos/heatmap_{number_of_vid}.png', heatmap_img)

    # Shot count
    frame_nums_with_ball_hits = ball_tracker.get_ball_shot_frames(ball_detections)
    shot_count = ball_tracker.get_shot_count_per_player(
        frame_nums_with_ball_hits, player_detections, ball_detections
    )
    print(f"✓ Shot count — Player 1: {shot_count[1]}, Player 2: {shot_count[2]}")

if __name__ == "__main__":
    main()