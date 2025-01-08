import cv2
import numpy as np
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera

# Motor GPIO setup
MOTOR_LEFT = 17
MOTOR_RIGHT = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_LEFT, GPIO.OUT)
GPIO.setup(MOTOR_RIGHT, GPIO.OUT)

def move_motors(direction):
    if direction == "left":
        GPIO.output(MOTOR_LEFT, GPIO.HIGH)
        GPIO.output(MOTOR_RIGHT, GPIO.LOW)
    elif direction == "right":
        GPIO.output(MOTOR_LEFT, GPIO.LOW)
        GPIO.output(MOTOR_RIGHT, GPIO.HIGH)
    elif direction == "forward":
        GPIO.output(MOTOR_LEFT, GPIO.HIGH)
        GPIO.output(MOTOR_RIGHT, GPIO.HIGH)
    else:
        GPIO.output(MOTOR_LEFT, GPIO.LOW)
        GPIO.output(MOTOR_RIGHT, GPIO.LOW)

# Load TFLite model
interpreter = Interpreter(model_path="tennis_ball_detector.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Camera setup
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
raw_capture = PiRGBArray(camera, size=(640, 480))

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    image = frame.array

    # Preprocess frame for TFLite model
    input_shape = input_details[0]['shape'][1:3]
    resized_frame = cv2.resize(image, (input_shape[1], input_shape[0]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    for i, score in enumerate(scores):
        if score > 0.5 and int(classes[i]) == 1:  # Assuming class ID 1 is for tennis ball
            box = boxes[i]
            ymin, xmin, ymax, xmax = box

            # Convert coordinates to pixel values
            height, width, _ = image.shape
            xmin, xmax = int(xmin * width), int(xmax * width)
            ymin, ymax = int(ymin * height), int(ymax * height)

            # Draw bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, "Tennis Ball", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Determine direction to move motors
            center_x = (xmin + xmax) // 2
            if center_x < width // 3:
                move_motors("left")
            elif center_x > 2 * width // 3:
                move_motors("right")
            else:
                move_motors("forward")
            break
    else:
        move_motors("stop")

    # Show frame
    cv2.imshow("Tennis Ball Detection", image)

    # Clear the stream for the next frame
    raw_capture.truncate(0)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
GPIO.cleanup()