# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
# from imutils.video import WebcamVideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    #grab the dimensions of frame, then initial blob
    #(h, w, c)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    #pass the blob through network and obtain face detection
    faceNet.setInput(blob)
    detections = faceNet.forward()

    #initialize lists of faces, their corresponding locations, and the list of predictions from face mask network
    faces = []
    locs = []
    preds = []

    #loop over the detections
    for i in range(0, detections.shape[2]):
        #extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        #filter out weak detections by ensuring the confidence is > min confidence
        if confidence > args["confidence"]:
            #compute (x, y) of bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #ensure the bounding box fall within the dimensions of frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))

            #extract face ROI
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            #TODO???
            face = img_to_array(face)
            #TODO???
            face = preprocess_input(face)

            #add the face and bounding boxes to lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

            #only make a prediction if at least one face was detected
            if len(faces)>0:
                #batch prediction on all faces
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)

            return (locs, preds)
#construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.7, help="min. probabilty to filter weak detections")
args = vars(ap.parse_args())

#load serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#load face mask detector model
print("[INFO] loading mask detector model...")
maskNet = load_model(args["model"])

#initialize the video stream and allow the camera sensor to warm up
print("[INFO] loading face mask detector model...")
vs = VideoStream(src=0).start()
# vs = WebcamVideoStream(src=0).start()
time.sleep(1.0)

#loop over the frames from the video stream
while True:
    #grab the frame from the thread video steam and resize it to have a max width 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #detect faces in the frame and determine if they are wearing a face mask or not
    # (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    result = detect_and_predict_mask(frame, faceNet, maskNet)
    if (result != None):
        (locs, preds) = result

        #loop over the detected face locations
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        #determine the class label and color, we'll use to draw the bounding box and text
        label = "Mask" if (mask - 0.5) > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        #include the prob in the label
        label = "{} : {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    #show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # key = cv2.waitKey(1)
    #if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()

