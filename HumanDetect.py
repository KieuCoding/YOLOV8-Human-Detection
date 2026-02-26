import cv2
import os
from ultralytics import YOLO


class PersonDetector:
    def __init__(
        self,
        model_path='yolov8n.pt',
        image_path='AnimeKirbyArtwork2.jpg',
        conf=0.3,
        camera_index=0
    ):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.conf = conf

        # Load image to display
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.display_image = cv2.imread(image_path)
        if self.display_image is None:
            raise ValueError("Failed to load image file")

        self.image_window_name = "Person Detected Image"
        self.image_shown = False

        # Camera
        self.cap = cv2.VideoCapture(camera_index)

    def show_image(self):
        cv2.imshow(self.image_window_name, self.display_image)
        self.image_shown = True

    def close_image(self):
        cv2.destroyWindow(self.image_window_name)
        self.image_shown = False

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run YOLO (PERSON ONLY)
            results = self.model(
                frame,
                imgsz=640,
                conf=self.conf,
                classes=[0]
            )

            boxes = results[0].boxes

            person_present = boxes is not None and len(boxes) > 0

            # Handle image display logic
            if person_present and not self.image_shown:
                print("👤 Person detected → showing image")
                self.show_image()

            elif not person_present and self.image_shown:
                print("🚫 No person → closing image")
                self.close_image()

            # Show YOLO annotated frame
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Person Detector", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = PersonDetector(
        model_path='yolov8n.pt',
        image_path='AnimeKirbyArtwork2.jpg',
        conf=0.3
    )
    detector.run()

