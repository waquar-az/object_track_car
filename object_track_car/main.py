import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")     #capture object to read frame from video

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=42)    #-2

while True:
    ret, frame = cap.read() 
    height, width, _ = frame.shape
     #print(height, width)
    #extract region of interset-------------------
    #-------------h,h+     ,w ,w+
    # Extract Region of interest
    roi = frame[340: 720,510: 800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)  #0-black,255-white, we want to remove all shadow expept white.so start from 254 to255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for i in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(i)
        if area > 100:
            #cv2.drawContours(roi, [i], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(i)


            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:           # Esc key to stop  
        break

cap.release()
cv2.destroyAllWindows()