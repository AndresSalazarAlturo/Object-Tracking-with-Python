#Utilities
import cv2 
import numpy as np
from object_detectionOpenCV import ObjectDetection
import math

#Initialize Object Detection
od = ObjectDetection()

#It's neccesary to import the video correctly to read the frames
cap = cv2.VideoCapture("D:/Proyectos_Python/SourceCode/source_code/los_angeles.mp4")

#Initialize count
count = 0

center_points_prev_frame = []

tracking_objects = {}
track_id = 0 

while(True):
    #Get the frames from de video
    #This line, take one frame from the video
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    
    #Point current frame
    center_points_cur_frame = []

    #Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        #print("FRAME N° ", count, " ", x , y, w, h)
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for pt in center_points_cur_frame:
        for pt2 in center_points_prev_frame:
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

            if distance < 20:
                tracking_objects[track_id] = pt
                track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)


    print("Tracking objects")
    print(tracking_objects)
    

    print("CUR FRAME")
    print(center_points_cur_frame)

    print("PREV FRAME")
    print(center_points_prev_frame)

    #Show the frames
    cv2.imshow("Frame", frame)

    #Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    #When use "0" in waitKey it stops in every single frame
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cap.destroyAllWindows()
