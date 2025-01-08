![](https://github.com/nilayarangil/nilayarangil.github.io/blob/main/assets/img/p7.png)

# CourtMate

CourtMate is an autonomous robot designed to revolutionize tennis court maintenance. Equipped with a Raspberry Pi and a suite of sensors, CourtMate utilizes OpenCV and TensorFlow Lite for intelligent navigation and object recognition. This innovative system efficiently sweeps courts, retrieves stray tennis balls, and enhances the overall playing experience.

# Key Components

## Hardware
![](https://github.com/nilayarangil/nilayarangil.github.io/blob/main/assets/img/p3.png)

1. Raspberry Pi: Serves as the primary control unit.
2. Pi Camera: Captures real-time video feed for processing.
3. Motors: Controlled using GPIO pins for directional movement.

## Software

1. TensorFlow Lite: Performs object detection using a pre-trained TFLite model.
2. OpenCV: Handles image preprocessing and visualization.
3. RPi.GPIO: Manages GPIO pin control for motors.

## Code Overview

### Motor Control

GPIO pins are set up to control two motors:

* Left Motor: Controlled via MOTOR_LEFT GPIO pin.
* Right Motor: Controlled via MOTOR_RIGHT GPIO pin.
```python
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
```

### TensorFlow Lite Model Integration

A pre-trained TFLite model (courtmate.tflite) is loaded and used for inference. The model detects objects and classifies them to identify tennis balls.
```python
interpreter = Interpreter(model_path="courtmate.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```
### Pi Camera Setup

The Pi Camera captures a continuous video stream for processing:
```python
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
raw_capture = PiRGBArray(camera, size=(640, 480))
```
### Image Processing and Inference

Each frame is resized and normalized before being passed to the TFLite model for inference.

```python
input_shape = input_details[0]['shape'][1:3]
resized_frame = cv2.resize(image, (input_shape[1], input_shape[0]))
input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
```

### Bounding Box and Motor Control

If a tennis ball is detected, the bounding box coordinates are used to determine its position in the frame. Motors are activated accordingly to move left, right, or forward.

```python
center_x = (xmin + xmax) // 2
if center_x < width // 3:
    move_motors("left")
elif center_x > 2 * width // 3:
    move_motors("right")
else:
    move_motors("forward")
```

### Cleanup

On exiting the program, resources are released:

```python
cv2.destroyAllWindows()
GPIO.cleanup()
```

### How to Run
![](https://github.com/nilayarangil/nilayarangil.github.io/blob/main/assets/img/p2.png)

* Connect the Pi Camera and motors to the Raspberry Pi.
* Install required libraries:
* pip install opencv-python tflite-runtime
* Place the pre-trained TFLite model (tennis_ball_detector.tflite) in the working directory.

Run the script:

* python courtmateTFPiCam.py

## Notes

* Ensure the motors are connected properly to avoid hardware damage.
* Adjust the confidence threshold (score > 0.5) and model class ID (int(classes[i]) == 1) as needed.

## Future Improvements

* Enhance the model to detect multiple objects simultaneously.
* Integrate additional sensors for better navigation.
* Optimize the motor control logic for smoother movements
