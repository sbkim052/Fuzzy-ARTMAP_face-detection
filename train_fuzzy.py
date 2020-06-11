# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:35:58 2020

@author: User
"""

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import numpy as np
import Artmap as fuz_art
 
# 성빈
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import os
import threading
import ctypes     

class Thread(threading.Thread):
    def _async_raise(self,tid, excobj):
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(excobj))
        if res == 0:
            raise ValueError("nonexistent thread id")
        elif res > 1:
            # """if it returns a number greater than one, you're in trouble, 
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
            raise SystemError("PyThreadState_SetAsyncExc failed")
    
    def raise_exc(self, excobj):
        assert self.isAlive(), "thread must be started"
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._async_raise(tid, excobj)
                return

        # the thread was alive when we entered the loop, but was not found 
        # in the dict, hence it must have been already terminated. should we raise
        # an exception here? silently ignore?

    def terminate(self):
        # must raise the SystemExit type, instead of a SystemExit() instance
        # due to a bug in PyThreadState_SetAsyncExc
        self.raise_exc(SystemExit)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
    help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to output label encoder")


#성빈
ap.add_argument("-d", "--detector", required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
args = vars(ap.parse_args())


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])


for i in range(len(data["embeddings"])):
    ma= max(data["embeddings"][i])
    mi=min(data["embeddings"][i])
    data["embeddings"][i] = (data["embeddings"][i]-mi)/(ma-mi)
#    print(max(data["embeddings"][i]), min(data["embeddings"][i]))
#    data["embeddings"][i]+=0.4

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")

#print(data["embeddings"][0:22][:])

A = fuz_art.Fuzzy_Artmap()
A.fit(np.array(data["embeddings"]),np.array(labels))




# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

#성빈
vs = VideoStream(src=0).start()
fps = FPS().start()



username=''
pred_list = []


while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    #input of the ARTMAP
    face_norm=''
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            face_resize = cv2.cvtColor(cv2.resize(face, (25,25), interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2GRAY)
            

            ma= max(face_resize.flatten())
            mi=min(face_resize.flatten())
            face_norm = (face_resize-mi)/(ma-mi)   
            #vec[0]=vec[0]+0.4
            
            

            
            j, index= A.predict(face_norm.flatten())
            
            
            if j!=None:
                # perform classification to recognize the face
                     
                name = le.classes_[j]
                pred_list.append(name)

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}".format(name)
            else:
                text = "Unknown"
                
                
                
                
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
              
    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    elif key == ord("r"):
        username=input("Enter your name(English) : ")
        A.register_flag=1
        le.classes_=np.concatenate((le.classes_,[username]))
        print(len(pred_list), pred_list.count("Sungbin"))
        pred_list=[]

    
    if A.register_flag!=0:
        t = Thread(target=A.register, args=(face_norm.flatten(),username))
        t.start()
        if A.register_flag==3:
            A.register_flag=0
            t.terminate()


        
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
