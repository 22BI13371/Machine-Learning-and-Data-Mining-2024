from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import glob
import joblib
import os
import numpy as np
import math
import random
import pickle
import random
from FeatureExtract import *


def distance(point1, point2):
    distance_tmp = math.sqrt(
        (point1.x-point2.x)*(point1.x-point2.x)+(point1.y-point2.y)*(point1.y-point2.y))
    return (distance_tmp)


LM = 1
width = 1440
height = 900


class Point:
    def __init__(self, x_init, y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])


D = 5  # number of scale
W = 8  # window size is 2W+1
Nh = 128
index = 0
num_correct = 0
num_prediction = 0
num_test = 2
num_random_point = 5000
# model = joblib.load("/storage/tonlh/Imorph/Model/ERT_uSub_3.sav")
minx = 400
miny = 400
maxx = 700
maxy = 700
model = joblib.load("Exercise2-3/logistic.sav")
for name in glob.glob("*.tif"):
    print(name)
    name_img = name
    input = []
    name_tmp = name[0:(len(name)-4)]
    name_tps = name_tmp+'.tps'
    f = open(name_tps)
    lines = f.readlines()
    xy = lines[9].split()
    point_true = Point(float(xy[0]), 1024-float(xy[1]))
    print(name)
    listImgs = RescaleImage(name, D)
    print(name)
    pointT = []
    for i in range(0, 500): 
        x_tmp = random.randint(minx, maxx)
        y_tmp = random.randint(miny, maxy)
        pointT.append(Point(x_tmp, y_tmp))
        listPoints = RescalePoint(x_tmp, y_tmp, D)
        w = computeRAW(listImgs, listPoints, W)
        input.append(w)
    output = model.predict(input)
    x_tmp = 0
    y_tmp = 0
    num_positive = 0
    for i in range(0, len(output)):
        if output[i] == 1:
            num_positive = num_positive+1
            # print(points[i])
            x_tmp = x_tmp+pointT[i].x
            y_tmp = y_tmp+pointT[i].y
    if num_positive != 0:
        num_prediction = num_prediction+1
        point_predict = Point(x_tmp/num_positive, y_tmp/num_positive)
        dis = distance(point_predict, point_true)
        print('error = ', dis)
    else:
        print('khong detect dc')
