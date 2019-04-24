import cv2
import argparse
import glob
import numpy as np
import os
import time
import sys
import numpy as np
import statistics
from PIL import Image, ImageDraw

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

from re3_utils.util import drawing
from re3_utils.util import bb_util
from re3_utils.util import im_util

from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT
from constants import PADDING

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
pts_trans = 0
time_0 = 0
canvas_np = np.zeros((3000, 639))
canvas_pil = Image.fromarray(canvas_np)
canvas_draw = ImageDraw.Draw(canvas_pil)

drawnBox = np.zeros(4)
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False
def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0,2]] = np.sort(boxToDraw[[0,2]])
    boxToDraw[[1,3]] = np.sort(boxToDraw[[1,3]])


def show_webcam(mirror=False):
    time0 = 0
    pts_trans = [0,0]
    speed_list = [0,0,0,0,0,0,0,0,0,0]
    global tracker, initialize, mouseupdown, boxToDraw
    cam = cv2.VideoCapture('video.mp4')
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    cv2.namedWindow('Perspective', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Perspective', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Canvas', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    print(OUTPUT_HEIGHT)
    print(OUTPUT_WIDTH)
    cv2.setMouseCallback('Original', on_mouse, 0)
    frameNum = 0
    outputDir = None
    paused = False
    while True:
        timer = time.time()
        interval = 1.0
        ret_val, img_og = cam.read()
        img_per = img_og
        #Perspecvtive Transform
        tl = [250, 90]
        tr = [430, 90]
        bl = [0, 359]
        br = [639, 359]
        
        pts1 = np.float32([tl, tr, bl, br])
        pts2 = np.float32([[0, 0], [639, 0], [0, 3000], [639, 3000]])
      
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_per = cv2.warpPerspective(img_per, matrix, (639, 3000))
        if img_og is None:
            # End of video.
            break
        if mirror:
            img_np = cv2.flip(img_np, 1)
        origImg = img_og.copy()
        drawImg = img_og.copy()
        origImg_per = img_per.copy()
        drawImg_per = img_per.copy()
        while frameNum == 0 or mousedown or paused:
            drawImg = img_og.copy()
            drawImg_per = img_per.copy()
            if mouseupdown:
                paused = False
                frameNum += 1
            cv2.rectangle(drawImg,
                    (int(boxToDraw[0]), int(boxToDraw[1])),
                    (int(boxToDraw[2]), int(boxToDraw[3])),
                    [0,0,255], PADDING)
            #cv2.circle(drawImg,(int((boxToDraw[0] + boxToDraw[2])/2),int((boxToDraw[1] + boxToDraw[3])/2)), 2, (0,0,255), -1)
            cv2.imshow('Original', drawImg)
            cv2.imshow('Perspective', drawImg_per)
            cv2.imshow('Canvas', np.array(canvas_pil))
            cv2.waitKey(2)

        if initialize:
            boxToDraw = tracker.track('webcam', img_og[:,:,::-1], boxToDraw)
            initialize = False
        else:
            boxToDraw = tracker.track('webcam', img_og[:,:,::-1])
        # cv2.rectangle(drawImg,
        #         (int(boxToDraw[0]), int(boxToDraw[1])),
        #         (int(boxToDraw[2]), int(boxToDraw[3])),
        #         [0,0,255], PADDING)
        cv2.circle(drawImg,(int((boxToDraw[0] + boxToDraw[2])/2),int((boxToDraw[1] + boxToDraw[3])/2)), 4, (0,0,255), -1)
        x_norm = int((boxToDraw[0] + boxToDraw[2])/2)
        y_norm = int((boxToDraw[1] + boxToDraw[3])/2)

        pts_norm = np.array([[x_norm,y_norm]], dtype = "float32")
        pts_norm = np.array([pts_norm])
        # pts_trans = pts_norm
        time1 = time0
        pts_trans_1 = pts_trans
        pxl_to_mtr = 1.5/88
        pts_trans = cv2.perspectiveTransform(pts_norm, matrix)
        pts_trans = np.squeeze(pts_trans).tolist()
        time0 = time.time()
        #x_0 = pts_trans[1]
        #print(pts_trans)
        x,y = pts_trans
        x1, y1 = pts_trans_1
        dx = (y-y1)*pxl_to_mtr
        dt = time0 - time1
        ref = 12
        speed = round(abs(dx/dt)*ref, 1)
        
        if speed:
            speed_list.pop()
            speed_list.insert(0, speed)
            # print(statistics.mean(speed_list))
        cv2.circle(drawImg_per,(int(pts_trans[0]),int(pts_trans[1])), 15, (0,0,255), -1)
        canvas_draw.ellipse([int(pts_trans[0]) - 2,int(pts_trans[1]) - 2, int(pts_trans[0]) + 2,int(pts_trans[1])+ 2], fill = 'red')
        # print([int(pts_trans[0]),int(pts_trans[1])])
        cv2.imshow('Original', drawImg)
        cv2.imshow('Perspective', drawImg_per)
        cv2.imshow('Canvas', np.array(canvas_pil))
        keyPressed = cv2.waitKey(2)
        if keyPressed == 27 or keyPressed == 1048603:
            break  # esc to quit
        elif keyPressed != -1:
            paused = True
            mouseupdown = False
        frameNum += 1
        while True:
            if time.time() - timer > interval:
                break

        speed = dx/interval
        print(speed)
    cv2.destroyAllWindows()



# Main function
if __name__ == '__main__':
    
    time0 = 0
    pts_trans = 0
    tracker = re3_tracker.Re3Tracker()

    show_webcam(mirror=False)
