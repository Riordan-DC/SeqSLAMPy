#! /usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

display = False

def loadFrames(directory, n):
    frames = [] 
    capture = cv2.VideoCapture(directory)
    count = 0
    while True:
        (grabbed, frame) = capture.read()
        if grabbed and count < n:
            #frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),                    (64, 32))
            frame = cv2.resize(frame, (64, 32), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
            count += 1
        else:
            break
            #cv2.imshow(directory, frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
    print("Frames from video {} loaded! size: {} kb".format(directory,
        sys.getsizeof(frames)/1024))
    return frames

def processFrames(frames):
    #Apply patch normalization and local contrast enhancement
    return applyLocalContrast(frames) 

def compareFrames(reference, query):
    pass

'''
    This normalizes a difference vector of a
    query/reference pair by normalizing it 
    against a window of surrounding vectors
'''
def normalizeWindow(window):
    pass

def buildDifferenceMatrix(refFrames, queryFrames):
    rows, width, height = np.shape(refFrames)
    pixels = width * height
    print("Width {} - Height {}".format(width, height))
    columns = len(queryFrames)
    window_height = window_width = 400
    if rows != columns:
        #Resize the matrix to fix in a window 
        window_height = 400
        window_width = columns
        if columns < 400:
            window_width = 400

    matrix = np.empty([rows, columns])
    print("Difference matrix of size: \n\t - ref {} \n\t - query {}".format(rows, columns))
    for row in range(rows):
        for column in range(columns):
            #D = np.true_divide(refFrames[row], (128)) - np.true_divide(queryFrames[column], (128))
            D = refFrames[row] - queryFrames[column]
            SAD = np.sum(np.absolute(D)) / pixels
            #print(SAD)
            matrix[row][column] = SAD
            if display:
                cv2.imshow('matrix', cv2.resize(matrix, (window_height, window_width), interpolation=cv2.INTER_AREA))
        if display:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if display:
        while True:
            cv2.imshow('matrix', cv2.resize(matrix, (window_height, window_width), interpolation=cv2.INTER_AREA))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    plt.imshow(matrix)
    plt.colorbar()
    plt.show()
    return matrix

def normaliseDifferenceMatrix(matrix, window):
    rows, columns = np.shape(matrix)
    window_height = window_width = 400
    if rows != columns:
        #Resize the matrix to fix in a window 
        window_height = 400
        window_width = columns
        if columns < 400:
            window_width = 400

    if rows%window != 0:
        #Window will leave remainders
        window += 1
    norm_matrix = np.zeros([rows, columns])
    for column in range(columns):
        for row in range(rows):
            #print(matrix[row:row+window, column])
            ya = int(max(0,row-(window/2)))
            yb = int(min(rows, row+(window/2)))
            #print("WINDOW: {} -{}- {}".format(ya,row,yb))
            local_vector = matrix[ya:yb, column]
            norm_matrix[row][column] = (matrix[row][column] - np.mean(local_vector)) / np.std(local_vector)
            if display:
                cv2.imshow('matrix', cv2.resize(matrix, (window_height, window_width), interpolation=cv2.INTER_AREA))
        if display:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if display:
        while True:
            cv2.imshow('matrix', cv2.resize(matrix, (window_height, window_width), interpolation=cv2.INTER_AREA))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    plt.imshow(norm_matrix)
    plt.colorbar()
    plt.show()
    return norm_matrix
            

def applyLocalContrast(frames):
    #This is incorrect, they do not actually increase contrast, they use patch normalization
    #which they state will enhance the contrast. Therefore the code below for the CLAHE contrast
    #algorithm is technically wrong. 
    #uses the CLAHE algorithm
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4,4))
    for index in range(len(frames)):
        frame = cv2.cvtColor(frames[index], cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(frame)
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        frames[index] = cv2.cvtColor(cv2.cvtColor(limg, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)/256
    '''
    window = 4
    width, height = np.shape(frames[0])
    print(width, height)
    for index in range(len(frames)):
        frame = frames[index]
        for m in range(0, height, window):
            for n in range(0, width, window):
                chunk = frame[m:m+window,n:n+window]
                print(np.shape(chunk))
                
    '''
    return frames 

def trajectoryScore(row, column, v_step, ds):
    pass

def getLine(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    x_coords = []
    y_coords = []
    for x in range(x1,x2):
        y = y1 + dy * (x - x1) // dx
        x_coords.append(x)
        y_coords.append(y)
    return x_coords,y_coords

def trajectorySearch(matrix):
    rows,columns = np.shape(matrix)
    score_matrix = np.stack((np.empty([rows, columns]),)*3, axis=-1)
    ds = 10
    v_steps = np.arange(0,3,1)
    display_matrix = np.stack((matrix,)*3, axis=-1)
    for column in range(columns):
        match_scores = []
        for row in range(rows):
            ds_ = int(ds/2)
            local_scores = [] #Chunk (match) local scores
            chunk = matrix[max(0,row-ds_):row+ds_,max(0,column-ds_):column+ds_]
            #This is where I construct a local score for multiple velocities
            #score = np.trace(chunk)
            for v_step in v_steps:
                x_line,y_line = getLine(0,0+v_step,np.shape(chunk)[0],np.shape(chunk)[1]-v_step)
                sequence = chunk[x_line, y_line]
                score = np.sum(sequence)
                local_scores.append(score)

            match_scores.append(min(local_scores))
            if True:
                cv2.imshow('trajectory search', score_matrix)

        best_score = np.argmin(match_scores)
        score_matrix[best_score][column] = 1
        #display_matrix = cv2.circle(display_matrix, (column, row), 1, (0,255,0), -1)
        #matrix = cv2.circle(matrix, (row, column), 3, (0,255,0), -1)
        #cv2.imshow('trajectory search', score_matrix)
        if True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    plt.imshow(score_matrix)
    plt.show()
    
if __name__ == "__main__":
    refFrames = processFrames(loadFrames('./day_trunc.avi', 300))
    queryFrames = processFrames(loadFrames('./night_trunc.avi', 300))
    matrix = normaliseDifferenceMatrix(buildDifferenceMatrix(refFrames[75:-5], queryFrames[85:]), 10)
    print(np.shape(matrix))  
    trajectorySearch(matrix)
    del refFrames[:]
    del queryFrames[:]
    cv2.destroyAllWindows()
