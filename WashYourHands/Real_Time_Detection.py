import tensorflow as tf
import cv2
import numpy as np

class_names = ["Not_Washing", "Washing"]
model_test = tf.keras.models.load_model('/Users/lukek/Wash_Your_GOD_DAMN_HANDS/WashYourHands/Wash_Your_GOD_DAMN_HANDS.h5')

cap = cv2.VideoCapture(r"C:\Users\lukek\Desktop\WashYourGODAMNHANDS\RawFootage\MVI_4059.MP4")

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized_img = cv2.resize(gray,(50,50))
    numpy_resized = np.array(resized_img).reshape(-1, 50, 50, 1)
    numpy_resized=numpy_resized/255
    pred = model_test.predict(numpy_resized)
    print(pred)
    pred_string = class_names[np.argmax(pred)]

    cv2.putText(frame, pred_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("frame", frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cv2.destroyAllWindows()