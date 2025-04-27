from robot_hat import Pin, Servo #type:ignore
import time
from ultralytics import YOLO #type:ignore
import numpy as np
from picamera2 import Picamera2, Preview #type:ignore

# Initialize pins for relay control
pin_0 = Pin("D0")
pin_1 = Pin("D1")

# Initialize YOLO model
model = YOLO('/home/pi/NightVisionFloodlight/best.pt')

def initialize_system():
    """Ensure relays are off initially."""
    pin_0.value(0)
    pin_1.value(0)
    time.sleep(1)

def process_detection_and_control():
    """Process live feed for object detection and control relays."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            frame = picam2.capture_array()

            # Convert frame to RGB if necessary
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]

            results = model(frame)  # Run YOLO model on the current frame

            object_detected = False

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Calculate center of the detected object
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2

                    object_detected = True

            if object_detected:
                while object_detected:
                    # Rapidly toggle between pin_0 and pin_1
                    pin_0.value(1)
                    pin_1.value(0)
                    time.sleep(0.1)
                    pin_0.value(0)
                    pin_1.value(1)
                    time.sleep(0.1)

                    # Check for continued object detection
                    frame = picam2.capture_array()
                    if frame.shape[-1] == 4:
                        frame = frame[:, :, :3]
                    results = model(frame)
                    object_detected = any(len(result.boxes) > 0 for result in results)
            else:
                # Turn off both pins if no object is detected
                pin_0.value(0)
                pin_1.value(0)

            time.sleep(0.1)  # Avoid excessive processing load
    finally:
        picam2.stop()
        pin_0.value(0)  # Ensure relay is off when exiting
        pin_1.value(0)

if __name__ == "__main__":
    # Initialize system state
    initialize_system()

    # Start the detection and control process
    process_detection_and_control()
