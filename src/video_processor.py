"""
Video processing pipeline for real-time parking detection
"""
import cv2
import time
from pathlib import Path
from detector import ParkingDetector


class VideoProcessor:
    def __init__(self, model_path='models/best.pt'):
        """
        Initialize video processor with parking detector
        
        Args:
            model_path: Path to trained YOLO model
        """
        self.detector = ParkingDetector(model_path)
        
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process video file and detect parking spaces
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, counts = self.detector.process_frame(frame)
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS to frame
            cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame
            if writer:
                writer.write(annotated)
            
            # Display frame
            if display:
                cv2.imshow('ParkVision - Parking Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {counts}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {frame_count / elapsed:.2f}")
    
    def process_webcam(self, camera_id=0):
        """
        Process live webcam feed
        
        Args:
            camera_id: Camera device ID (default: 0)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        print("Starting webcam feed. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, counts = self.detector.process_frame(frame)
            
            cv2.imshow('ParkVision - Live Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    processor = VideoProcessor('models/best.pt')
    
    # Process video file
    # processor.process_video('data/parking_video.mp4', 'output/result.mp4')
    
    # Or process webcam
    # processor.process_webcam(0)
