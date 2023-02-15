from queue import PriorityQueue
import cv2
import time
from InvidistModule import *

xv_in = [0,5,10,15,20,25,30,35,40,45,50,55,60,62,63,0,5,10,15,20,25,30,35,40,45,50,55,60,62,63,32]
yv_in = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,63,54,52,10,13,17,30,44,33,22,10,7,2,27,39,32]
values_in = [14,7,6,3,2,15,14,2,10,3,4,1,10,9,15,7,8,4,10,14,17,2,4,4,3,10,15,13,7,1,9]

risk_matrix_ = invDist(xv_in, yv_in, values_in, xsize=64, ysize=64, power=1, smoothing=20)

start_time = time.time()

Astar(64, risk_matrix_, 42,0,0,63 )

end_time = time.time()
#print(theWay)

img = cv2.imread("./Images/1.png")
for Point in theWay:
    img[Point[1]][Point[0]] = [255,0,0]

cv2.imwrite("./Path/Path_AStar.png",img)
    
risk =0
img = cv2.imread('./Path/Path_AStar.png')
for x in range(64):
    for y in range(64):
        if(img[y][x][0] == 255 and img[y][x][1] == 0 and img[y][x][2] == 0):
            risk += risk_matrix_[y][x]
print("AStar")
print("Time: ", end_time-start_time)
print("Risk:",risk)