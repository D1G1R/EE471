import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
from typing import Tuple, List


class ObjectDetector:
    
    def __init__(self, model_path: str, max_results: int = 5, 
                 score_threshold: float = 0.5, running_mode: str = 'IMAGE'):
        """
        Initialize the Object Detector.
        
        Args:
            model_path: Path to the TensorFlow Lite model file (.tflite)
            max_results: Maximum number of detection results to return
            score_threshold: Confidence threshold for detections
            running_mode: 'IMAGE', 'VIDEO', or 'LIVE_STREAM'
        """
        self.model_path = model_path
        self.max_results = max_results
        self.score_threshold = score_threshold
        self.running_mode = running_mode
        self.detector = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Create and initialize the ObjectDetector task."""
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Map string mode to VisionRunningMode
        mode_map = {
            'IMAGE': VisionRunningMode.IMAGE,
            'VIDEO': VisionRunningMode.VIDEO,
            'LIVE_STREAM': VisionRunningMode.LIVE_STREAM
        }
        
        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            max_results=self.max_results,
            score_threshold=self.score_threshold,
            running_mode=mode_map.get(self.running_mode, VisionRunningMode.IMAGE)
        )
        
        self.detector = vision.ObjectDetector.create_from_options(options)
    
    def detect_image(self, image_path: str) -> 'mp.tasks.vision.ObjectDetectorResult':
        """
        Perform object detection on a single image.
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            ObjectDetectorResult containing detected objects
        """
        # Load image from file
        mp_image = mp.Image.create_from_file(image_path)
        
        # Perform detection
        detection_result = self.detector.detect(mp_image)
        
        return detection_result
    
    def detect_numpy_array(self, image_array: 'np.ndarray') -> 'mp.tasks.vision.ObjectDetectorResult':
        """
        Perform object detection on a numpy array.
        
        Args:
            image_array: Input image as numpy array
            
        Returns:
            ObjectDetectorResult containing detected objects
        """
        # Convert numpy array to MediaPipe Image
        import numpy as np
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_array.astype(np.uint8)
        )
        
        # Perform detection
        detection_result = self.detector.detect(mp_image)
        
        return detection_result
    
    def visualize_detections(self, image_path: str, output_path: str = None):
        """
        Detect objects in an image and draw bounding boxes on it.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the annotated image (optional)
        """
        # Load image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Perform detection
        detection_result = self.detect_image(image_path)
        
        # Draw bounding boxes
        annotated_image = self._draw_detections(image, detection_result)
        
        # Display results
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to: {output_path}")
        
        return annotated_image
    
    def _draw_detections(self, image: 'np.ndarray', 
                        detection_result: 'mp.tasks.vision.ObjectDetectorResult') -> 'np.ndarray':
        """
        Draw detection results on the image.
        
        Args:
            image: Input image as numpy array
            detection_result: Detection results from the model
            
        Returns:
            Annotated image with bounding boxes and labels
        """
        import numpy as np
        
        annotated_image = image.copy()
        
        if detection_result.detections:
            for detection in detection_result.detections:
                # Get bounding box
                bbox = detection.bounding_box
                start_point = (int(bbox.origin_x), int(bbox.origin_y))
                end_point = (int(bbox.origin_x + bbox.width), 
                           int(bbox.origin_y + bbox.height))
                
                # Get category
                if detection.categories:
                    category = detection.categories[0]
                    category_name = category.category_name.capitalize()
                    
                    # Colors mapping loosely to the requested image style
                    if "dog" in category_name.lower():
                        bbox_color = (255, 204, 51) # Cyan/Light blue (BGR: 51, 204, 255)
                    elif "cat" in category_name.lower():
                        bbox_color = (255, 153, 102) # Cornflower blue
                    else:
                        bbox_color = (244, 212, 66) # Default cyan

                    # Draw thick bounding box rectangle
                    cv2.rectangle(annotated_image, start_point, end_point, bbox_color, 4)
                    
                    # Setup text label and sizes
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_thickness = 2
                    
                    text_size = cv2.getTextSize(category_name, font, font_scale, font_thickness)[0]
                    text_width, text_height = text_size[0], text_size[1]
                    text_padding = 10
                    
                    # Anchor point: Top-Right of the Bounding Box
                    x2 = end_point[0]
                    y1 = start_point[1]
                    
                    # Calculate filled rectangle coordinates for the background
                    rect_start = (x2 - text_width - text_padding * 2, y1)
                    # Text box height goes down inside the bbox
                    rect_end = (x2, y1 + text_height + text_padding * 2)
                    
                    # Draw a filled rectangle with the SAME color as the bounding box
                    cv2.rectangle(annotated_image, rect_start, rect_end, bbox_color, cv2.FILLED)
                    
                    # Write the class name inside the filled rectangle in BLACK
                    text_x = x2 - text_width - text_padding
                    text_y = y1 + text_height + text_padding
                    cv2.putText(annotated_image, category_name, (text_x, text_y), 
                              font, font_scale, (0, 0, 0), font_thickness)
        
        return annotated_image
    
    def print_detections(self, detection_result: 'mp.tasks.vision.ObjectDetectorResult'):
        """
        Print detection results to console.
        
        Args:
            detection_result: Detection results from the model
        """
        if not detection_result.detections:
            print("No objects detected.")
            return
        
        print(f"\nDetected {len(detection_result.detections)} object(s):\n")
        
        for i, detection in enumerate(detection_result.detections):
            print(f"Detection #{i}:")
            
            # Bounding box
            bbox = detection.bounding_box
            print(f"  Box: (x: {bbox.origin_x:.0f}, y: {bbox.origin_y:.0f}, "
                  f"w: {bbox.width:.0f}, h: {bbox.height:.0f})")
            
            # Categories and scores
            if detection.categories:
                print("  Categories:")
                for category in detection.categories:
                    print(f"    {category.category_name}: {category.score:.4f} "
                          f"(index: {category.index})")
            print()
    
    def close(self):
        """Close the detector and clean up resources."""
        if self.detector:
            self.detector.close()


def download_model(model_url: str = None, save_path: str = "model.tflite") -> str:
    """
    Download an object detection model if not already present.
    
    Args:
        model_url: URL to download the model from
        save_path: Path to save the model
        
    Returns:
        Path to the model file
    """
    import urllib.request
    
    if os.path.exists(save_path):
        print(f"Model already exists at: {save_path}")
        return save_path
    
    if model_url is None:
        # Default EfficientDet Lite 0 model from TensorFlow Hub
        model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
    
    print(f"Downloading model from: {model_url}")
    urllib.request.urlretrieve(model_url, save_path)
    print(f"Model saved to: {save_path}")
    
    return save_path


# Example usage
if __name__ == "__main__":
    """
    Example: Object detection on an image
    
    Before running this script:
    1. Install requirements: pip install mediapipe opencv-python numpy
    2. Have an image file ready for testing
    3. Optionally download a pre-trained model
    """
    
    # Model path (you can download from Google's TensorFlow Hub)
    # https://www.tensorflow.org/lite/examples/object_detection/overview
    model_path = os.path.join(os.path.dirname(__file__), "efficientdet_lite0.tflite")
    
    # Download model automatically if not exists
    if not os.path.exists(model_path):
        print("Model dosyası bulunamadı. İndiriliyor...")
        model_path = download_model(save_path=model_path)
    
    # Initialize detector
    detector = ObjectDetector(
        model_path=model_path,
        max_results=5,
        score_threshold=0.5,
        running_mode='IMAGE'
    )
    
    # Example: Detect objects in an image
    image_path = "4.jpg"  # Replace with your image
    
    if os.path.exists(image_path):
        print(f"Processing: {image_path}")
        
        # Perform detection
        results = detector.detect_image(image_path)
        
        # Print results
        detector.print_detections(results)
        
        # Visualize and save
        output_image = detector.visualize_detections(
            image_path,
            output_path="detections_output.jpg"
        )
        
        # Display with OpenCV (optional)
        cv2.imshow("Object Detection Results", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a test image to run the example.")
    
    # Clean up
    detector.close()