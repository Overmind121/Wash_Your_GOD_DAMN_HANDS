import cv2
import numpy as np
import os

file = 5

print("hello")

cap = cv2.VideoCapture(r"C:\Users\lukek\Desktop\WashYourGODAMNHANDS\Not_Washing_vid\Not_Washing_vid12.MP4")
path = r"C:\Users\lukek\Desktop\WashYourGODAMNHANDS\Not_Washing"


count = 10841

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    modified = cv2.resize(gray, (50, 50))
    cv2.imwrite(os.path.join(path, "not_wash_pic" + str(count) + ".jpg"), modified)
    cv2.imshow("cap", frame)
    count += 1
    print(count)
cap.release()
