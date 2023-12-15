import cv2
import os
import imutils
import numpy as np

path_to_video = r"./madhubala.mp4"
path_to_model = r"./model"
protoPath = os.path.sep.join([path_to_model, "colorization_deploy_v2.prototxt"])
modelPath = os.path.sep.join([path_to_model, "colorization_release_v2.caffemodel"])
pointPath = os.path.sep.join([path_to_model, "pts_in_hull.npy"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
cap = cv2.VideoCapture(path_to_video)
frame_counter = 0
writer = None

class8_ab = net.getLayerId("class8_ab")
conv8_313_rh = net.getLayerId("conv8_313_rh")
points = np.load(pointPath).transpose().reshape(2, 313, 1, 1)
net.getLayer(class8_ab).blobs = [points.astype("float32")]
net.getLayer(conv8_313_rh).blobs = [np.full([1, 313], 2.606, dtype="float32")]

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    frame = imutils.resize(frame, 500)
    Lab = cv2.cvtColor(frame.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
    Lab_resized = imutils.resize(Lab, width=640)
    L = cv2.split(Lab_resized)[0] - 50
    net.setInput(cv2.dnn.blobFromImage(L))
    a_b = net.forward()[0, :, :, :].transpose((1, 2, 0))
    a_b = cv2.resize(a_b, (frame.shape[1], frame.shape[0]))
    L = cv2.split(Lab)[0]
    updated_Lab = np.concatenate((L[:, :, np.newaxis], a_b), axis=2)
    output = cv2.cvtColor(updated_Lab, cv2.COLOR_LAB2BGR)
    output = (255 * np.clip(output, 0, 1)).astype("uint8")
    final_version = np.concatenate((frame, output), axis=1)

    # Get the dimensions of the current frame
    frame_height, frame_width, _ = frame.shape

    cv2.imshow("Result", final_version)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("color.vid", fourcc, 20, (frame_width * 2, frame_height), True)
    if writer is not None:
        writer.write(final_version)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

if writer is not None:
    writer.release()
cap.release()
cv2.destroyAllWindows()