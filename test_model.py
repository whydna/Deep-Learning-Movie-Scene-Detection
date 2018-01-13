# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video", required=True,
    help="path to the video file")
args = vars(ap.parse_args())

model = load_model(args["model"])

capture = cv2.VideoCapture(args["video"])

while True:
    # grab the current frame
    (grabbed, frame) = capture.read()

    # if we are viewing a video and did not a grab a frame then we have reached
    # the end of the video
    if args.get("video") and not grabbed:
        break

    # resize, convert to grayscale, and then clone it (so we can annotate it)
    frame = imutils.resize(frame, width=300)
    frameClone = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = gray.astype("float") / 255.0

    gray = img_to_array(gray)

    gray = np.expand_dims(gray, axis=0)

    (drinking, notDrinking) = model.predict(gray)[0]

    if drinking > notDrinking and drinking > 0.9:
        label = "Drinking: {:.2f}%".format(drinking * 100)
        cv2.putText(frameClone, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 2)

        cv2.imshow("Frame", frameClone)
        cv2.waitKey()


# clean up
camera.release()
cv2.destroyAllWindows()
