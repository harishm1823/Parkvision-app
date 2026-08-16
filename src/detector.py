"""
ParkVision - Parking Space Detector using YOLOv8
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class ParkingDetector:
    def __init__(self, model_path='models/best.pt', conf_threshold=0.5):
        """
        Initialize the parking detector with YOLOv8 model
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        try:
            # Try to load the custom model first
            self.model = YOLO(model_path)
            self.class_names = ['empty', 'occupied']
            print(f"✅ Loaded custom model: {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load custom model: {e}")
            print("🔄 Falling back to YOLOv8n pretrained model...")
            # Fallback to pretrained model for cars
            self.model = YOLO('yolov8n.pt')
            self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
            print("✅ Loaded YOLOv8n pretrained model")
        
    def detect(self, image):
        """
        Detect parking spaces in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            results: Detection results with bounding boxes and classes
        """
        results = self.model(image, conf=self.conf_threshold)[0]
        return results
    
    def count_spaces(self, results):
        """
        Count empty and occupied parking spaces
        
        Args:
            results: YOLO detection results
            
        Returns:
            dict: Counts of empty and occupied spaces
        """
        counts = {'empty': 0, 'occupied': 0, 'total': 0}
        
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls < len(self.class_names):
                    counts[self.class_names[cls]] += 1
                    counts['total'] += 1
        
        return counts
    
    def draw_detections(self, image, results):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image
            results: YOLO detection results
            
        Returns:
            annotated_image: Image with drawn detections
        """
        annotated = image.copy()
        
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Color: Green for empty, Red for occupied
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                label = f"{self.class_names[cls]}: {conf:.2f}"
                
                # Draw box and label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def process_frame(self, frame):
        """
        Process a single frame: detect, count, and annotate
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (annotated_frame, counts)
        """
        results = self.detect(frame)
        counts = self.count_spaces(results)
        annotated = self.draw_detections(frame, results)
        
        # Add count overlay
        text = f"Empty: {counts['empty']} | Occupied: {counts['occupied']} | Total: {counts['total']}"
        cv2.putText(annotated, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated, counts
