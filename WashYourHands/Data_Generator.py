import cv2
import numpy as np
import os

file = 5

print("hello")

cap = cv2.VideoCapture(r"C:\Users\lukek\Desktop\WashYourGODAMNHANDS\IMG_1821.MOV")
path = r"C:\Users\lukek\Desktop\WashYourGODAMNHANDS\Not_Washing"


count = 0

while cap.isOpened() and count <720:
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(path, "not_wash_pic" + str(count+3825) + ".jpg"), frame)
    cv2.imshow("cap", frame)
    count += 1
    print(count)
cap.release()
