# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:47:42 2018

@author: khalifazada
"""

import cv2

# facial, eye & smile cascade objects
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# detecting face + eyes function
# gray - B&W image
# frame - original image
def detect(gray, frame):
  # gray - b&w image
  # 1.3 - scaling factor
  # 5 - minimum number of neighbors
  faces = face_cascade.detectMultiScale(gray, 1.3, 10)
  
  # detect a face
  for (x, y, w, h) in faces:
    # draw a rectangle on the color image
    # frame - draw on this image
    # (x, y) - upper-left corner coordinate
    # (x+w, y+h) - lower-right corner coordinate
    # (255, 0, 0) - RGB color fo the rectangle
    # 2 - thickness of edges of the rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # zone of interest (i.e the area of the image inside the rectangle)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    
    # detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 20)
    for (ex, ey, ew, eh) in eyes:
      # draw rectangle around eyes
      cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # detect smile
    smile = smile_cascade.detectMultiScale(roi_gray, 1.6, 50)
    for (sx, sy, sw, sh) in smile:
      # draw rectangle around smile
      cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
  # return original image with overlaid rectangle
  return frame

# detecting faces + eyes from webcam video
# 0 - webcam of computer (1 - external webcam)
video_capture = cv2.VideoCapture(0)
# repeat capture and detection infinitely (untel break)
while True:
  # ignore first element returned
  _, frame = video_capture.read() # last frame of the webcam
  # modify frame to make it b&w
  # frame - webcam feed
  # cv2.COLOR_BGR2GRAY - average greyscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # canvas - result of the detect function
  canvas = detect(gray, frame)
  # display modified images
  # Video - 
  # canvas - image with detected rectangles
  cv2.imshow('Video', canvas)
  # detect 'Q' key pressed to break loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
# turn off the webcam
video_capture.release()
# destroy all windows
cv2.destroyAllWindows()