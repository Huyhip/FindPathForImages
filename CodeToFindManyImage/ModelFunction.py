# from threading import currentThread
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from keras.preprocessing import image
import csv
# from math import sqrt
# from math import pow

model = keras.models.load_model('./model_1_4_new')

# Round function


def roundTo(a, maxsize):
    if maxsize == 8:
        if a == 0 or a == 1:
            a = 0
        if a == 2 or a == 3:
            a = maxsize/3
        if a == 4 or a == 5:
            a = 2 * maxsize/3
        if a == 6 or a == 7:
            a = maxsize - 1
        return a

    else:
        if a < maxsize/6:
            a = 0
        elif a >= maxsize/6 and a < maxsize / 2:
            a = maxsize/3
        elif a >= maxsize/2 and a < 5*maxsize / 6:
            a = 2 * maxsize/3
        else:
            a = maxsize - 1
        return a

# Predict and draw the basic line


def predictAndDrawBasicLine(imgInput, tuplePoints):
    (beginX, beginY, endX, endY) = tuplePoints
    img = imgInput.copy()
    # Determine begin and end point
    beginPoint = [beginX, beginY]
    endPoint = [endX, endY]
    (h, d, w) = img.shape
    if h > 4:
        for i in range(0, 2):
            beginPoint[i] = roundTo(beginPoint[i], h)
            endPoint[i] = roundTo(endPoint[i], h)
        coordinate = {"00": [0, 0],
                    "01": [0, d/3],
                    "02": [0, 2*d/3],
                    "03": [0, d-1],
                    "10": [h/3, 0],
                    "11": [h/3, d/3],
                    "12": [h/3, 2*d/3],
                    "13": [h/3, d-1],
                    "20": [2*h/3, 0],
                    "21": [2*h/3, d/3],
                    "22": [2*h/3, 2*d/3],
                    "23": [2*h/3, d-1],
                    "30": [h-1, 0],
                    "31": [h-1, d/3],
                    "32": [h-1, 2*d/3],
                    "33": [h-1, d-1]}

    if h == 4:
        coordinate = {"00": [0, 0],
                    "01": [0, 1],
                    "02": [0, 2],
                    "03": [0, 3],
                    "10": [1, 0],
                    "11": [1, 1],
                    "12": [1, 2],
                    "13": [1, 3],
                    "20": [2, 0],
                    "21": [2, 1],
                    "22": [2, 2],
                    "23": [2, 3],
                    "30": [3, 0],
                    "31": [3, 1],
                    "32": [3, 2],
                    "33": [3, 3]}
    begin = "00"
    end = "00"

    for point in coordinate:
        if coordinate[point] == beginPoint:
            begin = point
        if coordinate[point] == endPoint:
            end = point

    if begin != end:
        imgResize = cv2.resize(img, (4, 4))
        imgHSV = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)

        # Tạo mảng lower và upper nhằm mục đích sử dụng cho hàm inRange
        lower = np.array([31, 0, 0])
        upper = np.array([255, 255, 255])

        mark = cv2.inRange(imgHSV, lower, upper)

        # Chuyển về màu ban đầu
        imgResult = cv2.bitwise_and(imgResize, imgResize, mask=mark)
        imgResult = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)

        imgResize = tf.keras.utils.img_to_array(imgResult)
        imgResize = imgResize/255.
        imgResize = np.array([imgResize])

        #predict_result = model.predict(imgResize)

        # opening the csv file
        with open('./data2.csv') as csv_file:
            # reading the csv file using DictReader
            csv_reader = csv.DictReader(csv_file)
            # converting the file to dictionary
            # by first converting to list
            # and then converting the list to dict
            dict_from_csv = dict(list(csv_reader)[0])
            # making a list from the keys of the dict
            list_of_column_names = list(dict_from_csv.keys())
            # displaying the list of column names
            # print("List of column names : ",list_of_column_names)

        predictList = []

        predict_result = model.predict(imgResize)
        for i in range(112):
            if predict_result[0][i] > 0.8:
                # print(predict_result[0][i])
                predictList.append(list_of_column_names[i].split('-'))

        # Draw a line in img, prioritize path have 3 point
        haveLine = False

        for path in predictList:
            if len(path) == 3 and begin in path and end in path:
                cv2.line(img, (int(coordinate[path[0]][0]), int(coordinate[path[0]][1])), (int(coordinate[path[1]][0]), int(coordinate[path[1]][1])), (255, 0, 0), 1)
                cv2.line(img, (int(coordinate[path[1]][0]), int(coordinate[path[1]][1])), (int(coordinate[path[2]][0]), int(coordinate[path[2]][1])), (255, 0, 0), 1)
                haveLine = True
                break
        if haveLine == False:
            cv2.line(img, (int(coordinate[begin][0]), int(coordinate[begin][1])), (int(coordinate[end][0]), int(coordinate[end][1])), (255, 0, 0), 1)
    return (img, (int(beginPoint[0]), int(beginPoint[1]), int(endPoint[0]), int(endPoint[1])))
# Convert Point
def defineRegion(h , point):
    (x , y) = point
    if (x < h/2):
        xRegion = "Left"
    else:
        xRegion = "Right"
    if (y < h/2):
        yRegion = "Up"
    else:
        yRegion = "Down"
    
    if (xRegion == "Left" and yRegion == "Up"):
        return "img0"
    elif (xRegion == "Right" and yRegion == "Up"):
        return "img2"
    elif (xRegion == "Left" and yRegion == "Down"):
        return "img1"
    else:
        return "img3"

#backTracing @Overwrite
def backTracking(imgInput, tuplePoints):
    (beginX, beginY, endX, endY) = tuplePoints
    imgRaw = imgInput.copy()
    (h, d, w) = imgInput.shape
    if( h == 16 and d == 16):
        return predictAndDrawBasicLine(imgInput, tuplePoints)[0]
    else:
        (imgInput, tuplePoints) = predictAndDrawBasicLine(imgInput, tuplePoints)
        (beginX, beginY, endX, endY) = tuplePoints
        part = {0: [0, int(h/2), 0, int(d/2)],
                1: [int(h/2), int(h), 0, int(d/2)],
                2: [0, int(h/2), int(d/2), int(d)],
                3: [int(h/2), int(h), int(d/2), int(d)]}
        img0 = imgRaw[part[0][0]:part[0][1], part[0][2]: part[0][3]]
        img1 = imgRaw[part[1][0]:part[1][1], part[1][2]: part[1][3]]
        img2 = imgRaw[part[2][0]:part[2][1], part[2][2]: part[2][3]]
        img3 = imgRaw[part[3][0]:part[3][1], part[3][2]: part[3][3]]
        
        listPoints= {"img0" :[], "img1" :[], "img2" :[], "img3" :[]}
        listPoints[defineRegion(h, (beginX,beginY))].append((int(beginX%(d/2)), int(beginY%(h/2))))
        listPoints[defineRegion(h, (endX,endY))].append((int(endX%(d/2)), int(endY%(h/2))))
        # find by x-axis
        for i in range(0, d, 1):
            if all(imgInput[int(h/2)][i] == (255, 0, 0)):
                regionA = defineRegion(h , (i,int(h/2)))
                regionB = defineRegion(h , (i,int(h/2-1)))
                if i == d/2:
                    xA = int((d/2-1)%(d/2))
                    yA = int((h/2-1)%(h/2))
                    xB = int(i%(d/2))
                    yB = int((h/2)%(h/2))
                    listPoints["img0"].append((xA, yA))
                    listPoints[regionA].append((xB, yB))
                    break
                else :
                    xA = int(i%(d/2))
                    yA = int((h/2)%(h/2))
                    xB = int(i%(d/2))
                    yB = int((h/2-1)%(h/2))
                    listPoints[regionA].append((xA, yA))
                    listPoints[regionB].append((xB, yB))
                    break
        # find by y-axis
        for i in range(0, h, 1):
            if all(imgInput[i][int(d/2)] == (255, 0, 0)):
                regionA = defineRegion(h , (int(d/2),i))
                regionB = defineRegion(h , (int(d/2-1),i))
                if i == h/2:
                    break
                else :
                    xA  = int((d/2)%(d/2))
                    yA = int(i%(h/2))
                    xB  = int((d/2-1)%(d/2))
                    yB = int(i%(h/2))
                    listPoints[regionA].append((xA, yA))
                    listPoints[regionB].append((xB, yB))
                    break

        # listPoints = sorted(listPoints,key=lambda d: d['y'])
        listPoints["img0"].sort(key=lambda tup: tup[1])
        listPoints["img1"].sort(key=lambda tup: tup[1])
        listPoints["img2"].sort(key=lambda tup: tup[1])
        listPoints["img3"].sort(key=lambda tup: tup[1])
        if (len(listPoints["img0"]) == 2):
            (xA , yA) = listPoints["img0"][0]
            (xB , yB) = listPoints["img0"][1]
            pointInput = (xA, yA, xB, yB)
            imgInput[part[0][0]:part[0][1], part[0][2]: part[0][3]] = backTracking(img0, pointInput)
        if (len(listPoints["img1"]) == 2):
            (xA , yA) = listPoints["img1"][0]
            (xB , yB) = listPoints["img1"][1]
            pointInput = (xA, yA, xB, yB)
            imgInput[part[1][0]:part[1][1], part[1][2]: part[1][3]] = backTracking(img1, pointInput)
        if (len(listPoints["img2"]) == 2):
            (xA , yA) = listPoints["img2"][0]
            (xB , yB) = listPoints["img2"][1]
            pointInput = (xA, yA, xB, yB)
            imgInput[part[2][0]:part[2][1], part[2][2]: part[2][3]] = backTracking(img2, pointInput)
        if (len(listPoints["img3"]) == 2):
            (xA , yA) = listPoints["img3"][0]
            (xB , yB) = listPoints["img3"][1]
            pointInput = (xA, yA, xB, yB)
            imgInput[part[3][0]:part[3][1], part[3][2]: part[3][3]] = backTracking(img3, pointInput)
    return imgInput


        