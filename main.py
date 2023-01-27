import cv2
import numpy as np
import time

# load yolo models
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# load classes list and image
classes = []
with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()

# cap = cv2.VideoCapture(0) # open camera
cap = cv2.VideoCapture("./video test.mp4")

while True:

  start = time.time()

  _, frame = cap.read()

  height, width, _ = frame.shape

  # create input blob and perform forward pass
  blob = cv2.dnn.blobFromImage(
      frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
  net.setInput(blob)
  output_layers_names = net.getUnconnectedOutLayersNames()
  layersOutputs = net.forward(output_layers_names)

  boxes = []
  confidances = []
  class_ids = []

  for output in layersOutputs:
    for detection in output:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidance = scores[class_id]

      if confidance > 0.5:
        center_x = int(detection[0]*width)
        center_y = int(detection[1]*height)
        w = int(detection[2]*width)
        h = int(detection[3]*height)

        x = int(center_x - w/2)
        y = int(center_y - h/2)

        boxes.append([x, y, w, h])
        confidances.append((float(confidance)))
        class_ids.append(class_id)

  indexes = cv2.dnn.NMSBoxes(boxes, confidances, 0.5, 0.4)

  font = cv2.FONT_HERSHEY_PLAIN
  colors = np.random.uniform(0, 255, size=(len(boxes), 3))

  # get fps of the videos predictions
  end = time.time()
  total_time = end - start
  fps = 1 / total_time

  # except if no object found
  try:
    for i in indexes.flatten():
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      color = colors[i]
      confidance = str(round(confidances[i], 2))

      # draw bounding box and write label and FPS on frame
      cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
      cv2.putText(frame, label + " " + confidance,
                  (x, y+20), font, 2, (0, 0, 0), 2)
      cv2.putText(frame, "FPS: " + str( round(fps) ), (10, 50), font, 2, (0, 0, 0), 2)

      print(label)
      if label in ["car", "truck"]:
        print("do something ...")
        """

        == using pyfirmata ==
        rotate servo to open 
        
        """
      print("rotate servo to close ")


  except AttributeError:
      pass

  cv2.imshow("Video", frame)

  # kllik esc stop program
  key = cv2.waitKey(1)
  if key == 27:
      break

cap.release()
cv2.destroyAllWindows()