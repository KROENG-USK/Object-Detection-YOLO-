"""
ALGORITMA KAPAL AUTONOM UNTUK MENDETEKSI SAMPAH
"""

import cv2
import numpy as np
import math
# load yolo models
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# load classes list and image
classes = []
with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(1)

while True:
  _, img = cap.read()
  height, width, _ = img.shape
 

  center_frame = (int(width/2), int(height/2))
  cv2.circle(img, center_frame, 5, (0, 255, 0), -1 )                 # center frame

  # create input blob and perform forward pass
  blob = cv2.dnn.blobFromImage(
      img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
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

  #print(indexes.flatten())

  data_class = {}
  dir = ''

  # except error if no object found
  try :
    for i in indexes.flatten():
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      color = colors[i]
      confidance = str(round(confidances[i], 2))

      # draw bounding box and label on image
      cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
      cv2.putText(img, label + " " + confidance,
                  (x, y+20), font, 2, (0, 0 , 0), 2)
      # draw center of frame on image 
      cv2.circle(img, (x + int(w/2), y + int(h/2)), 5, (0, 255, 0), -1 ) # center img detected
      
      # height_2 =       

      center_img_detect = (x + int(w/2), y + int(h/2))
      self_point = (int(width/2), height)

      cv2.line(img, (center_frame[0], center_frame[1]), (self_point[0], self_point[1]), (255, 0, 0), 2)
      cv2.line(img, (center_img_detect[0], center_img_detect[1]), (self_point[0], self_point[1]), (255, 0, 255), 2)

# R = jarak dari titik self_point ke center_img_detect
      R = height - center_img_detect[1]
      S = center_img_detect[0] - int(width /2)

      error = (math.atan(S/R) * 180 / math.pi)

      servo_putar = 90 - error

      if center_img_detect[0] > width/2:
        dir = 'kanan'

      elif center_img_detect[0] < width/2:
        dir = 'kiri' 
 
     
      print(f"""
      Sudut Servo {servo_putar} 
      direction {dir}""")
       
      #print(error)  # convert radian to degree
      data_class[str(i)] = error
    
    print(data_class)
      #print(S, R, error)
      # print(center_img_detect, center_frame, self_point)
    
  except AttributeError:
      pass

  cv2.imshow("Image", img)

  key = cv2.waitKey(1)
  if key == 27:
      break

cap,release()
cv2.destroyAllWindows()
