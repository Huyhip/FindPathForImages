from math import pow
from math import sqrt
import numpy as np
from matplotlib import cm
import cv2

def pointValue_in(x,y,z,power,xv,yv,zv,values):
    nominator=0
    denominator=0
    for i in range(0,len(values)):
        dist = sqrt((x-xv[i])*(x-xv[i])+(y-yv[i])*(y-yv[i])+(z-zv[i])*(z-zv[i])+5*5)
        if(dist<0.0000000001):
            return values[i]
        nominator=nominator+(values[i]/pow(dist,power))
        denominator=denominator+(1/pow(dist,power))
    if denominator > 0:
        value = nominator/denominator
    else:
        value = -9999
    return value

def pointValue_out(x,y,z,power,xv,yv,zv,values):
    nominator=0
    denominator=0
    c=1.113195e5
    for i in range(0,len(values)):
        dist = sqrt((c*x-c*xv[i])**2+(c*y-c*yv[i])**2+(z-zv[i])**2)
        if(dist<0.0000000001):
            return values[i]
        nominator=nominator+(values[i]/pow(dist,power))
        denominator=denominator+(1/pow(dist,power))
    if denominator > 0:
        value = nominator/denominator
    else:
        value = -9999
    return value

def invDist_in(xv,yv,zv,values,zi,xsize=100,ysize=100,power=1):
    xi = np.linspace(0,xsize-1 , xsize)
    yi = np.linspace(0,ysize-1, ysize)
    valuesGrid = np.zeros((ysize,xsize))
    for y in range(0,ysize):
        for x in range(0,xsize):
            valuesGrid[y][x] = pointValue_in(xi[x],yi[y],zi,power,xv,yv,zv,values)
    return valuesGrid


def invDist_out(xv,yv,zv,values,zi,xsize=100,ysize=100,power=1):
    xi = np.linspace(0,xsize-1 , xsize)
    yi = np.linspace(0,ysize-1, ysize)
    valuesGrid = np.zeros((ysize,xsize))
    for y in range(0,ysize):
        for x in range(0,xsize):
            valuesGrid[y][x] = pointValue_out(xi[x],yi[y],zi,power,xv,yv,zv,values)
    return valuesGrid

def NormalizeData(data,max_col):
    return data / max_col

def NormalizeData_smooth(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def ConvertRange(old_value,old_min,old_max,new_min,new_max):
    return ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

def save(layer,max_col, zi):
    #normalize matrix to (0,1)
    # normalized_matrix =  NormalizeData(layer,max_col)
    normalized_matrix = (layer-np.min(layer))/(np.max(layer)-np.min(layer))
    #print(normalized_matrix)

    #convert from range(0,1) to range (0.5 , 0.9)
    convert_to_range_05_09 = ConvertRange(normalized_matrix,0,1,0.5,0.9)
    #print(convert_to_range_05_09)

    viridis_big = cm.get_cmap('nipy_spectral')
    print("----------")

    #get the value of 'nipy_spectral' color
    matrix_color = viridis_big(convert_to_range_05_09)

    #remove the 4th values which is = 1
    matrix_color2 = matrix_color[:,:,:3]

    #multiple matrix with 255 to get the color RGB value matrix
    matrix_color3 = matrix_color2*255

    #write the matrix to an image
    cv2.imwrite('Images/Img' + str(zi) +'.png', matrix_color3)

    #cv2.imwrite
    #read image
    img = cv2.imread('Images/Img' + str(zi) +'.png')

    #convert image from RGB to BGR
    matrix_color4 = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    #save the image
    cv2.imwrite('Images/Img' + str(zi) +'.png', matrix_color4)

def point(img, xsize=512, ysize=256,yr=9.15,xr=4.25):
    x=[]
    y=[]
    for j in range(0,ysize):
        for i in range(0,xsize,10):
            if(img[j,i] == [255,0,0]).all():
                y.append(i)
                x.append(j)
    x = np.array(x)
    y = np.array(y)

    y_in = ConvertRange(NormalizeData(y, xsize), 0, 1, 0, yr)
    x_in = ConvertRange(NormalizeData(x, ysize), 0, 1, 0, xr)

    print(x_in)
    print(y_in)

def pointValue(x,y,power,smoothing,xv,yv,values):
    nominator=0
    denominator=0
    for i in range(0,len(values)):
        dist = sqrt((x-xv[i])*(x-xv[i])+(y-yv[i])*(y-yv[i])+smoothing*smoothing)
        #If the point is really close to one of the data points, return the data point value to avoid singularities
        if(dist<0.0000000001):
            return values[i]
        nominator=nominator+(values[i]/pow(dist,power))
        denominator=denominator+(1/pow(dist,power))

    #Return NODATA if the denominator is zero
    if denominator > 0:
        value = nominator/denominator
    else:
        value = -9999
    return value

def invDist(xv,yv,values,xsize=100,ysize=100,power=2,smoothing=0):
    valuesGrid = np.zeros((ysize,xsize))
    for x in range(0,xsize):
        for y in range(0,ysize):
            valuesGrid[y][x] = pointValue(x,y,power,smoothing,xv,yv,values)
    return valuesGrid

theWay = []

def ConvertList(way):
    l = []
    for i in range(len(way)-1, -1,-1):
        l.append(way[i])
    return l

def FindTheWay(beginX, beginY, endX, endY, Close):
    way = [(endX,endY)]
    while True: 
        if way[-1] == (beginX, beginY):
            return ConvertList(way)
            
        for Point in Close:
            if Point[0][0] == way[-1][0] and Point[0][1] == way[-1][1]:
                way.append((Point[1][0], Point[1][1]))
                break
        
#Define class PriorityQueue for Astar
class PriorityQueue():
    def __init__(self):
        self.queue = []
  
    def __str__(self):
        return ' ,'.join([str(i) for i in self.queue])
  
    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
    
    def FindOut(self, a):
        for i in range(len(self.queue)):
            if a == self.queue[i][0]:
                return True
        return False
    
    def top(self):
        return self.queue[0]
  
    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)
  
    # for popping an element based on Priority
    def delete(self, dic,Hx):
        try:
            max = 0
            for i in range(len(self.queue)):
                if abs(dic[self.queue[i][0]]) < abs(dic[self.queue[max][0]]):
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()

def Astar(size, matrixHx, beginX, beginY, endX, endY):
    
    matrixGx = []
    for i in range(size): 
        l = []
        for j in range(size): 
            l.append(0)
        matrixGx.append(l)
    ''' for i in range(size):
       for j in range(size):
            matrixGx[i][j] = (abs(beginX-j) + abs(beginY-i)) * 0.001 '''
    # print("\nMatrixGx: ")    
    # for i in range(size):
    #     print(matrixGx[i])
    
    #Declare matrixFx:
    matrixFx = []
    for i in range(size+1): 
        l = []
        for j in range(size+1): 
            l.append(0)
        matrixFx.append(l)
    for i in range(size):
        for j in range(size):
            matrixFx[i][j] = (matrixGx[i][j] + matrixHx[i][j])
        
    
    l = []
    for i in range(size):
        for j in range(size):
            l.append([(i,j, matrixFx[j][i]), matrixFx[j][i]])
    data = dict(l)
    
    #print(data[(beginX, beginY)])
    #print(data[(endX, endY)])

    #Declare Open, close queue
    Open = PriorityQueue()
    Close = PriorityQueue()
    Open.insert( [( beginX, beginY, matrixFx[beginY][beginX] ) , (-1,-1, 0) ] )
    while(True):
        if Open.isEmpty == True:
            print("Tim kiem that bai")
            return
        
        v = Open.delete(data, matrixHx)
        Close.insert(v)
        
        ##print("Duyet: ", v, data[v[0]])
        
        if v[0][0] == endX and v[0][1] == endY:
            print("Tim kiem thanh cong")
            Way = FindTheWay(beginX, beginY, endX, endY, Close.queue)
            for Point in Way:
                theWay.append(Point)
            return
        
        
        nearPoints = [(v[0][0]+1, v[0][1], matrixFx[ v[0][1] ][ v[0][0]+1 ]),
                      (v[0][0]+1, v[0][1]+1, matrixFx[v[0][1]+1][ v[0][0]+1]),
                      (v[0][0], v[0][1]+1, matrixFx[v[0][1]+1][ v[0][0]]), 
                      (v[0][0]-1, v[0][1]+1, matrixFx[v[0][1]+1][ v[0][0]-1]), 
                      (v[0][0]-1, v[0][1], matrixFx[v[0][1]][ v[0][0]-1])]
        for Point in nearPoints:
            if Point[0] < 0 or Point[1] < 0 or Point[0] >= size or Point[1] >= size or (Close.FindOut(Point) and (data[v[0]] + Point[2]) >= data[Point]):
                continue
            if Open.FindOut(Point) and (data[v[0]] + Point[2]) < data[Point]: 
                for openPoint in Open.queue:
                    if openPoint[0] == Point:
                        openPoint[1] = v[0]
                        data[openPoint] = data[v[0]] + Point[2]
                        continue
            if Open.FindOut(Point) and (data[v[0]] + Point[2]) >= data[Point]: 
                continue
            data[Point] = data[v[0]] + Point[2];
            Open.insert([Point, v[0]])