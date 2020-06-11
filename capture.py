import cv2
import sys
import time
import os
import pickle
import numpy as np


class Face():
    def __init__(self, frame_id, name, box, encoding):
        self.frame_id = frame_id
        self.name = name
        self.box = box
        self.encoding = encoding


class Faceregistering():
    def __init__(self):
        self.faces_id = []
        self.faces =[]
        self.run_encoding = False
        self.capture_dir = "captures"

    def capture_filename(self, frame_id):
        return "%d.jpg" % frame_id


    def encode(self, src_file, capture_per_second, stop=0, r_name=""):
        src = cv2.VideoCapture(src_file)
        if not src.isOpened():
            return

        self.faces =[]
        self.faces_id = []
        frame_id = 0
        frame_r_name = "%s" % r_name
        frame_rate = src.get(5)
        stop_at_frame = int(stop * frame_rate)
        frames_between_capture = int(round(frame_rate) / capture_per_second)

        print("start encoding from src: %dx%d, %f frame/sec" % (src.get(3), src.get(4), frame_rate))
        print(" - capture every %d frame" % frames_between_capture)
        if stop_at_frame > 0:
            print(" - stop after %d frame" % stop_at_frame)

        cascPath = sys.argv[1]
        faceCascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
        #eyeEascade = cv2.CascadeClassifier('D:\opencv\data\haarcascades\haarcascade_eye.xml')
        #faceCascade = cv2.CascadeClassifier(cascPath)

        while True:
            # Capture frame-by-frame
            ret, frame = src.read()

            if frame is None:
                break

            frame_id += 1
            if frame_id % frames_between_capture != 0:
                continue

            if stop_at_frame > 0 and frame_id > stop_at_frame:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            pathname = os.path.join(self.capture_dir,
                                    self.capture_filename(frame_id))
            cv2.imwrite(pathname, frame)
            self.faces.extend(frame)
            self.faces_id.append(frame_id)

            time.sleep(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        src.release()
        cv2.destroyAllWindows()


    def register(self, r_name):
        if len(self.faces) is 0:
            print("no faces to register")
            return


        dir_name = "./dataset/%s" % r_name
        os.mkdir(dir_name)

        indexes = len(self.faces_id)

        for i in range(indexes) :
            frame_id = self.faces_id[i]
            print(frame_id)
            pathname = os.path.join(self.capture_dir,
                        self.capture_filename(frame_id))
            image = cv2.imread(pathname)
            filename = r_name + "-" + self.capture_filename(frame_id)
            #filename = dir_name + str(counter)
            pathname = os.path.join(dir_name, filename)
            cv2.imwrite(pathname, image)
            #shutil.copy2(pathname, 'knowns')

        print("%s registered to the dataset" %r_name)

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self.faces))

    def load(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
            self.faces = pickle.loads(data)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encode",
                    help="video file to encode or '0' to encode web cam")
    ap.add_argument("-c", "--capture", default=1, type=int,
                    help="# of frame to capture per second")
    ap.add_argument("-s", "--stop", default=0, type=int,
                    help="stop encoding after # seconds")
    ap.add_argument("-i", "--name", type=str, required=True,
                    help="enter registerer's name")
    args = ap.parse_args()

    fc = Faceregistering()

    if args.encode:
        src_file = args.encode
        if src_file == "0":
            src_file = 0
        fc.encode(src_file, args.capture, args.stop, args.name)
        fc.save("encodings.pickle")

    try:
        fc.load("encodings.pickle")
    except FileNotFoundError:
        print("No or invalid encoding file. Encode first using -e flag.")
        exit(1)
    fc.register(args.name)