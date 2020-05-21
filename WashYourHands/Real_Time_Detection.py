import tensorflow as tf
import cv2
import numpy as np
import serial
from nanpy import (ArduinoApi, SerialManager)
import time

print("Hello World")
try:
    connection = SerialManager()
    a = ArduinoApi(connection = connection)
except:
    print("Failed")
    
a.pinMode(13, a.OUTPUT)
a.pinMode(12, a.OUTPUT)
a.pinMode(11, a.OUTPUT)
a.pinMode(10, a.OUTPUT)
a.pinMode(9, a.OUTPUT)
a.digitalWrite(9, a.HIGH)

warning = 0
is_washing = False
isnt_washing = False
blink = 0
clean = False

class_names = ["Not_Washing", "Washing"]

model_test = tf.keras.models.load_model('/home/pi/Wash_Your_GOD_DAMN_HANDS/WashYourHands/Wash_Your_GOD_DAMN_HANDS.h5')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray,(50,50))
    numpy_resized = np.array(resized_img).reshape(-1, 50, 50, 1)
    numpy_resized=numpy_resized/255
    pred = model_test.predict(numpy_resized)
    print(pred)
    pred_string = class_names[np.argmax(pred)]
    
    if(pred_string == "Washing"):
        is_washing = True
        a.digitalWrite(9, a.LOW)
    elif(pred_string == "Not_Washing" and is_washing == True):
        isnt_washing = True
        is_washing = False
        
    if(is_washing == True and clean == False):
        warning = 0
        a.digitalWrite(13, a.LOW)
        a.digitalWrite(11, a.LOW)
        a.digitalWrite(10, a.HIGH)
        time.sleep(.5)
        a.digitalWrite(10, a.LOW)
        time.sleep(.5)
        blink+=1
        if(blink == 20):
            clean = True
            
    elif(isnt_washing == True and clean == False):
        a.digitalWrite(11, a.HIGH)
        time.sleep(.5)
        a.digitalWrite(11, a.LOW)
        time.sleep(.5)
        warning +=1
        if(warning == 3):
            a.digitalWrite(11, a.HIGH)
            a.digitalWrite(13, a.HIGH)
    
    elif(clean == True):
            a.digitalWrite(11, a.LOW)
            a.digitalWrite(10, a.LOW)
            a.digitalWrite(9, a.LOW)
            a.digitalWrite(12, a.HIGH)
            time.sleep(5)
            a.digitalWrite(12, a.LOW)
            a.digitalWrite(9, a.HIGH)
            clean = False
            
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        a.digitalWrite(13, a.LOW)
        a.digitalWrite(12, a.LOW)
        a.digitalWrite(11, a.LOW)
        a.digitalWrite(10, a.LOW)
        a.digitalWrite(9, a.LOW)
        break
    
    #cv2.putText(gray_display, pred_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    #cv2.imshow("frame", gray_display)
cv2.destroyAllWindows()
