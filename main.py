from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2

def main():
    number_of_vid= 3
    input_video_path = f'input_videos/inp_vid{number_of_vid}.mp4'
    video_frames = read_video(input_video_path)
    # Detect players and balls in the video frames
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolov8_best_50e.pt')
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path=f'tracker_stubs/player_detections_{number_of_vid}.pkl')

    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path=f"tracker_stubs/ball_detections_{number_of_vid}.pkl"
                                                     )
    
    #Court keypoints for a standard tennis court
    court_model_path = 'models/keypoints_model_04.pth'
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #choose and filter players based on court keypoints
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)


    #Minicourt 
    mini_court = MiniCourt(video_frames[0])

    #Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                           ball_detections, 
                                                                                                           court_keypoints)

    # Draw bounding boxes on the video frames
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #draw court keypoints on the video frames
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    
    # Draw mini court on the video frames
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))  
                                                               

    #Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the output video
    save_video(output_video_frames, f'output_videos/output_video_{number_of_vid}.avi')

if __name__ == "__main__":
    main()