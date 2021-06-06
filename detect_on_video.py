# Loading Libraries
import cv2                  # For Reading and Displaying Frames
import mediapipe            # For In-Built Face Detector Model
import numpy as np

# Loading Tensorflow Functions
# Function for loading saved model <filename.model>
from tensorflow.keras.models import load_model
# Function for transforming PIL Image to Numpy Array
from tensorflow.keras.preprocessing.image import img_to_array
# Preprocessing Function for Input Image before passing into MobileNet Model (Mask Detector Model)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Loading Pretrained Mask Detector Model from file mask_detector.model
# mask_detector.model is built from MOBILENET using Transfer Learning
mask_detector = load_model("mask_detector.model")

# Mediapipe's LightWeight Face Detection Model
face_detector = mediapipe.solutions.face_detection.FaceDetection()

# Initializing OpenCV's Video Caputure to get live feed from Camera
capture = cv2.VideoCapture(0)

# Looping over frames of Camera Feed
while True:

    # Get input frame from capture instance
    _, img = capture.read()

    # Convert BGR frame read by capture instance into RGB frame
    # Face Detection model expects RGB Frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pass RGB frame to Face Detector Model and get multiple detections of faces if exists
    results = face_detector.process(imgRGB)
    if results.detections:
        # Looping over each detection in RGB Frame
        for detection in results.detections:

            # Get relative bounding box of that detection
            boxR = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            
            # Get Absolute Bounding Box Positions
            # (startX, startY) - Top Left Corner of Bounding Box
            # (endX, endY)     - Bottom Right Corner of Bounding Box
            (startX, startY, endX, endY) = (boxR.xmin, boxR.ymin, boxR.width, boxR.height) * np.array([iw, ih, iw, ih])
            startX = max(0, int(startX))
            startY = max(0, int(startY))
            endX = min(iw - 1, int(startX + endX))
            endY = min(ih - 1, int(startY + endY))

            # Extracting the face from the RGB Frame to pass into Mask Detection Model
            face = imgRGB[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.array([face], dtype='float32')

            # Predicting Mask or No Mask on the extracted RGB Face
            preds = mask_detector.predict(face, batch_size=32)[0][0]
            label = "Mask" if preds < 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Drawing Bounding Box and Putting Text on the BGR frame
            cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

    # Displaying the Modified BGR Frame with Prediction onto opencv window
    cv2.imshow("Video", img)
    cv2.waitKey(1)
