
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Tạo model architecture
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.model.fc.in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 14 * 2)
        )

        # self.model.fc = torch.nn.Sequential(
        #     torch.nn.Linear(self.model.fc.in_features, 512),
        #     torch.nn.BatchNorm1d(512),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #     torch.nn.Linear(512, 14 * 2)
        # )

        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set to evaluation mode
        
        # Transform pipeline (giống như lúc validation)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """
        Dự đoán keypoints từ một frame
        Args:
            image: BGR image từ OpenCV
        Returns:
            keypoints: numpy array shape (28,) - 14 điểm x 2 tọa độ
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform và thêm batch dimension
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Convert to numpy và scale về kích thước ảnh gốc
        keypoints = outputs.squeeze().cpu().numpy()
        
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0   # Scale x coordinates
        keypoints[1::2] *= original_h / 224.0  # Scale y coordinates

        return keypoints

    def draw_keypoints(self, image, keypoints):
        """
        Vẽ keypoints lên ảnh
        Args:
            image: BGR image
            keypoints: numpy array shape (28,)
        Returns:
            image với keypoints đã vẽ
        """
        output_image = image.copy()
        
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            
            # Vẽ điểm tròn với viền
            cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)  # Điểm màu đỏ
            # cv2.circle(output_image, (x, y), 8, (255, 255, 255), 2)  # Viền trắng
            
            # Vẽ số thứ tự điểm
            cv2.putText(output_image, str(i//2 + 1), (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return output_image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Vẽ keypoints lên tất cả các frame của video
        Args:
            video_frames: list các frame
            keypoints: numpy array shape (28,) - keypoints cố định cho toàn bộ video
        Returns:
            output_video_frames: list các frame đã vẽ keypoints
        """
        output_video_frames = []
        for frame in video_frames:
            frame_with_keypoints = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame_with_keypoints)
        return output_video_frames
