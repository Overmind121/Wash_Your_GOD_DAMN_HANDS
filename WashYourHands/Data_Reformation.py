import cv2
import os
import pickle
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 50
DIR = r"C:\Users\lukek\Desktop\WashYourGODAMNHANDS"
categories = ["Not_Washing", "Washing"]
training_data = []
X = []
y = []
def reformat_imgs(dir):
    for cat in categories:
        path = os.path.join(dir, cat)
        print(cat)
        if cat == "Washing":
             for img in os.listdir(path):
                 path2 = os.path.join(path,img)
                 print(path2)
                 pic = cv2.imread(path2)
                 gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                 modified = cv2.resize(gray, (IMG_SIZE,IMG_SIZE))
                 cv2.imwrite(path2, modified)



def create_training_data():
    for cat in categories:
        path = os.path.join(DIR, cat)
        class_num = categories.index(cat)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

            training_data.append([img_arr, class_num])

create_training_data()
print(training_data[:1])
random.shuffle(training_data)
for samp in training_data[:1]:
    print(samp[1])

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.asarray(y, dtype=np.float32)
print(y[0])
print("Hello World")

with open("X.pickle", "wb") as pick:
    pickle.dump(X, pick)


with open("y.pickle", "wb") as pick:
    pickle.dump(y, pick)