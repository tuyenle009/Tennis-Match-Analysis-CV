import cv2
from ultralytics import YOLO
import numpy as np

def test_yolo_detection(video_path, model_path='yolov8x.pt', output_path='output_test.mp4'):
    """
    Test YOLO detection trên video để kiểm tra số lượng cầu thủ được detect
    
    Args:
        video_path: đường dẫn đến video input
        model_path: đường dẫn đến model YOLO
        output_path: đường dẫn lưu video output
    """
    # Load model
    model = YOLO(model_path)
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tạo VideoWriter để lưu kết quả
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    print("Bắt đầu detection...")
    
    frame_count = 0
    person_stats = []  # Thống kê số người detect được mỗi frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect với YOLO
        results = model.track(frame, persist=True, classes=[0], verbose=False)  # class 0 = person
        
        # Lấy thông tin detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Lấy track IDs nếu có
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = list(range(len(boxes)))
            
            person_stats.append(len(boxes))
            
            # Vẽ bounding boxes
            for i, (box, conf, track_id) in enumerate(zip(boxes, confidences, track_ids)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Màu khác nhau cho mỗi ID
                colors = [
                    (255, 0, 0),    # Xanh dương
                    (0, 255, 0),    # Xanh lá
                    (0, 0, 255),    # Đỏ
                    (255, 255, 0),  # Cyan
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Vàng
                ]
                color = colors[track_id % len(colors)]
                
                # Vẽ box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ label với ID và confidence
                label = f"ID:{track_id} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Hiển thị tổng số người detect
            info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len(boxes)}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            person_stats.append(0)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames} | Persons: 0", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Ghi frame vào video output
        out.write(frame)
        
        # Hiển thị progress
        if frame_count % 30 == 0:
            print(f"Processed: {frame_count}/{total_frames} frames")
    
    # Giải phóng resources
    cap.release()
    out.release()
    
    # In thống kê
    print("\n" + "="*50)
    print("THỐNG KÊ DETECTION:")
    print("="*50)
    if person_stats:
        print(f"Số người detect trung bình: {np.mean(person_stats):.2f}")
        print(f"Số người detect tối thiểu: {np.min(person_stats)}")
        print(f"Số người detect tối đa: {np.max(person_stats)}")
        print(f"Số frame detect được ít nhất 1 người: {sum(1 for x in person_stats if x > 0)}/{len(person_stats)}")
        print(f"Số frame detect được 2 người: {sum(1 for x in person_stats if x == 2)}/{len(person_stats)}")
        print(f"Số frame detect được >2 người: {sum(1 for x in person_stats if x > 2)}/{len(person_stats)}")
    
    print(f"\nVideo đã được lưu tại: {output_path}")
    print("Kiểm tra video để xem có detect nhầm không!")


def test_with_confidence_threshold(video_path, model_path='yolov8x.pt', conf_threshold=0.5):
    """
    Test với confidence threshold để filter detection
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    print(f"\nTest với confidence threshold = {conf_threshold}")
    print("="*50)
    
    frame_count = 0
    person_counts = []
    
    # Test 100 frames đầu
    while frame_count < 100 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect với confidence threshold
        results = model(frame, conf=conf_threshold, classes=[0], verbose=False)
        
        if results[0].boxes is not None:
            num_persons = len(results[0].boxes)
            person_counts.append(num_persons)
    
    cap.release()
    
    if person_counts:
        print(f"Trong {frame_count} frames:")
        print(f"  - Trung bình: {np.mean(person_counts):.2f} người")
        print(f"  - Min: {np.min(person_counts)}, Max: {np.max(person_counts)}")
        print(f"  - Số frame có 2 người: {sum(1 for x in person_counts if x == 2)}")
        print(f"  - Số frame có >2 người: {sum(1 for x in person_counts if x > 2)}")


if __name__ == "__main__":
    # Thay đổi đường dẫn video của bạn
    VIDEO_PATH = 'input_videos/input_video1.mp4'
    MODEL_PATH = "yolov8x.pt"
    OUTPUT_PATH = "output_videos/output_detection_test.mp4"
    
    # Test 1: Detection với tracking
    print("TEST 1: YOLO Detection + Tracking")
    test_yolo_detection(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH)
    
    # Test 2: Thử với các confidence threshold khác nhau
    # print("\n\nTEST 2: Thử các confidence threshold khác nhau")
    # for conf in [0.3, 0.5, 0.7]:
    #     test_with_confidence_threshold(VIDEO_PATH, MODEL_PATH, conf)