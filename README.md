# Real Time Face Mask Detection

### Built using Tensorflow, Mediapipe, OpenCV, Streamlit using Deep Learning concepts to detect face mask in real time video stream

<hr>

## Tech Stack Used

1. Tensorflow (Building Mask Detection Model on Face Images)
1. MediaPipe (To extract faces from Images)
1. OpenCV (To capture live feed and process it to models)
1. Streamlit (For building website to display detections)

<hr>

## Working

Frames from live video feed is extracted using OpenCV and passed into Mediapipe's Face Detector Model which extracts face from the frame. Next the extracted image is passed into pretrained Mask Detection Model which detects whether or not the person in the extracted image is wearing a mask.
The Mask Detection model is built using transfer learning on MobileNet. The predictions are drawn on to the frame and displayed in the website.

Here both MobileNet and Mediapipe Face Detector are light weight models which makes the detections on live video feed smooth and computationally efficient.

<hr>

## Streamlit App

*$ pip install -r requirements.txt*

*$ streamlit run app.py*
