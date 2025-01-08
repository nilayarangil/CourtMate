# CourtMate
CourtMate is an autonomous robot designed to revolutionize tennis court maintenance.

Equipped with a Raspberry Pi and a suite of sensors, CourtMate utilizes TensorFlow Lite for intelligent navigation and object recognition. This innovative system efficiently sweeps courts, retrieves stray tennis balls, and enhances the overall playing experience.

# Key Notes
Model Training: Replace "CourtMate.tflite" with the path to your pre-trained TensorFlow Lite model capable of detecting tennis balls.
Motor Control: Adjust GPIO pins and motor control logic as needed for your specific hardware setup.

# Dependencies
Install TensorFlow Lite runtime: pip install tflite-runtime.
Install OpenCV: pip install opencv-python.
Set up GPIO with RPi.GPIO.
