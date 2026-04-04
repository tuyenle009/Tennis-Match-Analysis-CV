from ultralytics import YOLO
import cv2
from utils import read_video, save_video


# Load video
video_frames = read_video('input_videos/inp_vid6.mp4')

# Load model
model = YOLO('models/yolov8m_players.pt')

output_frames = []
for frame in video_frames:
    results = model.predict(frame, conf=0.3, verbose=False)[0]
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        conf = box.conf.tolist()[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player1 {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    output_frames.append(frame)

save_video(output_frames, 'output_videos/test_pure_detection.avi')
print("✅ Done!")