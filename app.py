# Loading Libraries
import cv2 as cv            # For Reading and Displaying Frames
import mediapipe            # For In-Built Face Detector Model
import numpy as np
from PIL import Image

# Python Library for Building Website
import streamlit as st

# Loading Tensorflow Functions
# Function for loading saved model <filename.model>
from tensorflow.keras.models import load_model
# Function for transforming PIL Image to Numpy Array
from tensorflow.keras.preprocessing.image import img_to_array
# Preprocessing Function for Input Image before passing into MobileNet Model (Mask Detector Model)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Website Main Title
st.title("Face Mask Detection")

# Sidebar Select Box for Face Mask Detection on Image or Video
page = st.sidebar.selectbox("Image/ Live Video", ("Image", "Live Video"))

# Hiding Streamlit Footer and Menu
hide_footer = """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
         """
st.markdown(hide_footer, unsafe_allow_html=True)

# Loading Pretrained Mask Detector Model from file mask_detector.model
# mask_detector.model is built from MOBILENET using Transfer Learning
mask_detector = load_model("mask_detector.model")

# Mediapipe's LightWeight Face Detection Model
face_detector = mediapipe.solutions.face_detection.FaceDetection()

# If Image is selected in selectbox
if page == "Image":

    st.subheader("Detection on Image")

    # Get Image using File Uploader
    file = st.file_uploader("Upload Image", type='jpg')

    # If file is uploaded
    if file is not None:

        # Convert Input Image into PILLOW RGB Image
        file = Image.open(file).convert('RGB')
        file.save("images/in.jpg")
        img = cv.imread('images/in.jpg')
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.image(imgRGB, "Image Uploaded Successfully")

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
                face = cv.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.array([face], dtype='float32')

                # Predicting Mask or No Mask on the extracted RGB Face
                preds = mask_detector.predict(face, batch_size=32)[0][0]
                label = "Mask" if preds < 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # Drawing Bounding Box and Putting Text on the BGR frame
                cv.putText(img, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv.rectangle(img, (startX, startY), (endX, endY), color, 2)

        # Displaying the Modified BGR Frame with Prediction onto Website
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.image(img, 'Detections on Uploaded Image')

# If Video is Selected in Selectbox
else:

    st.subheader("Detection on Video")
    st.write("\n\n")

    # Checkbox to start video capture and display predictions
    run = st.checkbox("Select Checkbox to run Live Video and Uncheck to stop")

    # Start the Video Capture
    capture = cv.VideoCapture(0)

    # Initialize Image Widget to display frames of Video
    frame = st.image([])

    # When checkbox is selected start detection and prediction on frames
    while run:

        # Get input frame from capture instance
        _, img = capture.read()

        # Convert BGR frame read by capture instance into RGB frame
        # Face Detection model expects RGB Frame
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
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
                face = cv.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.array([face], dtype='float32')

                # Predicting Mask or No Mask on the extracted RGB Face
                preds = mask_detector.predict(face, batch_size=32)[0][0]
                label = "Mask" if preds < 0.5 else "No Mask"
                percentage = (1 - preds) * 100 if label == "Mask" else preds * 100
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label += f': {percentage:.2f}%'

                # Drawing Bounding Box and Putting Text on the BGR frame
                cv.putText(img, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv.rectangle(img, (startX, startY), (endX, endY), color, 2)

        # Displaying the Modified BGR Frame with Prediction onto opencv window
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        frame.image(img)

    # If stopped release Capturing
    else:
        capture.release()

