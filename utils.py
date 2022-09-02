try:
    import qi
    from naoqi import ALProxy
except:
    print('Not on real Robot')
import argparse
import sys
import time
from PIL import Image
import numpy as np
import cv2 as cv
import copy
import ffmpeg
import math
from PIL import ImageFont, ImageDraw, Image

import zmq
import json, ast
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    #import ipdb;ipdb.set_trace()
    #buf = memoryview(msg)
    #something wrong with md key name -> u'key1'
    A = np.frombuffer(msg, dtype=np.float32)
    return A.reshape(4)

def detect_keypoint_yolo(frame):

    global socket
    send_array(socket, frame)
    keypoint= recv_array(socket)
    
    if keypoint[0]==-100:
        return frame, None, None
    confidence = keypoint
    center = copy.deepcopy(keypoint[:2])
    center[1]=center[1] * frame.shape[0]
    center[0]=center[0] * frame.shape[1]
    #print(center)
    r = int(abs(keypoint[-1])*frame.shape[0])
    return frame,center.astype(np.int), r

def detect_keypoint(frame):
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    l_red_lower = np.array([2,150,71])
    l_red_upper = np.array([23,255,1715])
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #import ipdb;ipdb.set_trace()
    #u_red_lower = np.array([0,170,120])
    #u_red_upper = np.array([35,255,245])

    erosion_size=1
    erosion_shape=cv.MORPH_RECT
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)).astype(np.uint8)

    floodflags = 8
    floodflags |= cv.FLOODFILL_FIXED_RANGE
    previous_track = None
    blurred = cv.GaussianBlur(frame, (3,3), 0)
    img_hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    img_gray = cv.cvtColor(img_hsv, cv.COLOR_BGR2GRAY)
    
    #img_gray = cv.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    ret,thresh1 = cv.threshold(img_gray,131,255,cv.THRESH_BINARY)
    #green_mask = cv.inRange(img_hsv, low_green, high_green)
    #green_mask = 255-green_mask
    mask=cv.inRange(img_hsv, l_red_lower, l_red_upper)
    
    
    #if i==9:
        #import ipdb;ipdb.set_trace()
    #mask_u=cv.inRange(img_hsv, u_red_lower, u_red_upper)
    #mask = cv.Canny(blurred,100,200)#mask_l #+ mask_u
    
    tmp = mask.copy()
    #mask = cv.erode(mask.astype(np.uint8), kernel, iterations=2)
    
    mask_ = np.zeros((frame.shape[0], frame.shape[1]),np.uint8)
    contour = [] 
    contours,hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    radius=0
    if len(contours)>0:
        contours = contours[0]
        #print('contour',contour)
        #print('ctr', contours)
        (x,y), radius = cv.minEnclosingCircle(contours)
        center = (int(x), int(y))
        radius = int(radius)
    else:
        center=None
        radius = None

    
    cv.imwrite('test.jpg', img_gray)
    cv.imwrite('mask.jpg', mask)
    
    return frame, center, radius


def load_imgs(int1=1, int2=100, parent_folder='./datasets/rgb', name='farbe'):
    loaded = []
    for i in range(int1, int2):
        #import ipdb;ipdb.set_trace()
        loaded.append(cv.imread(parent_folder+'/'+name+str(i)+'.jpg'))
    return loaded

def convert_centers2out(c, tmp_c):
    #see where at which pixel the ball will roll out
    #d = c-tmp_c -> movement vector
    #scalar where the movement vecot hits image border S= (image_height - C.x)/d.x
    #scale d.y and add to current ball location C.y + d.y *scale
    dy = c[0]-tmp_c[0]
    dx = c[1]-tmp_c[1]

    #x is down
    if dx==0:
        dx+=1e-4
    scalar = (600-c[1])/( dx)
    print(scalar)
    if scalar<0:
        scalar=1e9
    out_pixel = c[0]+(dy * scalar)
    #print('s',scalar,' cutofy', cut_of_y,'d ', (dx,dy), ' C', c )
    d = np.array([dx,dy])
    d = d / np.linalg.norm(d)
    angle = np.arctan2(d[1],d[0])
    angle = angle*180/math.pi
    left_right = angle
    
    norm = np.sqrt((dx)**2 + (dy)**2)

    return out_pixel, angle, dx, dy, norm

def output_pixel2decision(cut_of_y, r):
    #cut_of_y = output pixel
    #r radius
    angle_ = '------'
   
    if cut_of_y>=320 and cut_of_y<=470:
        angle_='middle'
    elif cut_of_y>470 and cut_of_y<2000:
        angle_='right'
    elif cut_of_y<320 and cut_of_y>-2000:
        angle_='left'
    return angle_