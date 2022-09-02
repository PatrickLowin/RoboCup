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

#  Socket to talk to server
print("Connecting to hello world server...")
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
    elif cut_of_y>470 and cut_of_y<1000:
        angle_='right'
    elif cut_of_y<320 and cut_of_y>-300:
        angle_='left'
    return angle_

def goalkeeper(IP, PORT):
    motion = ALProxy("ALMotion", IP, PORT)
    video_service = ALProxy("ALVideoDevice",IP, PORT)
    tracker = ALProxy("ALTracker", IP, PORT) 
    motion = ALProxy("ALMotion", IP, PORT)
    posture = ALProxy("ALRobotPosture", IP, PORT)
    tts = ALProxy("ALTextToSpeech", IP, PORT)
    tts.say('connected')
    resolution = 2    # VGA
    colorSpace = 11   # RGB
    motion.wakeUp()

    fractionMaxSpeed = 0.8
    # Go to posture stand
    posture.goToPosture("Crouch", fractionMaxSpeed)
    videoClient = video_service.subscribe("python_client", resolution, colorSpace, 30)
    
    
    targetName = "RedBall"
    diameterOfBall = 0.06
    tracker.registerTarget(targetName, diameterOfBall)

    # set mode
    mode = "Head"
    tracker.setMode(mode)

    # Then, start tracker.
    
    tracker.track(targetName)
    t0 = time.time()
    counter =0
    ball_moving = True
    error_margin = 0.1
    counter = 0
    non_moving_frames = 5
    tmp_coordinates = tracker.getTargetPosition()
    #first phase find the ball
    #tts.say('toggleSearch')
    #search_on = True
    #tracker.toggleSearch(True)
    
    while True:
        #tracker.track(targetName)
        t0 = time.time()
        counter =0
        ball_moving = True
        error_margin = 0.1
        counter = 0
        non_moving_frames = 5
        tmp_coordinates = tracker.getTargetPosition()
        #first phase find the ball
        tts.say('toggleSearch')
        search_on = True
        #tracker.toggleSearch(True)

        while search_on:
            print(tracker.isTargetLost())
            print('still in search loop')
            print(tracker.getTargetPosition())
            #tracker.toggleSearch(True)
            if tracker.isNewTargetDetected():
                search_on=False

        while ball_moving:
            if len(tracker.getTargetPosition())==0 or len(tmp_coordinates)==0:
                tmp_coordinates=tracker.getTargetPosition()
                continue

            print(np.array(tracker.getTargetPosition()) - np.array(tmp_coordinates))
            if np.linalg.norm(np.array(tracker.getTargetPosition()) - np.array(tmp_coordinates))< error_margin:
                print('counter +1')
                counter += 1
            else:
                counter = 0
                tts.say('target lost')
            if counter == non_moving_frames:
                ball_moving = False
                tracker.stopTracker()

            tmp_coordinates = tracker.getTargetPosition()


        tts.say('I located the ball')
        tts.say('Expecting a shooting attempt')


        shooting_condition= True
        tmp_c = None
        angle=9000
        angle_='N'
        movement_threshold = 3
        visualize = True
        while shooting_condition:

            naoImage = video_service.getImageRemote(videoClient)
            
            counter +=1

            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_string = str(bytearray(array))
            img = np.fromstring(image_string, np.uint8).reshape([imageHeight, imageWidth, 3])
            img = cv.resize(img, (800,600))
            f,c,r = detect_keypoint_yolo(img)
            print(img.shape, c,r)
            
            if tmp_c is None:
                tmp_c = c
            elif tmp_c is not None and c is not None:
                out_pixel, angle, dx, dy, norm =convert_centers2out(c, tmp_c)
                print('out,angle, dx,dy,norm',out_pixel, angle, dx, dy, norm )
                decision = output_pixel2decision(out_pixel, r)
                print(decision)
                if movement_threshold < norm and decision!='N':  
                    shooting_condition = False
                    tts.say(decision)
                    break

                if visualize:
                    #f = cv.resize(f, (800,600))
                    f = cv.arrowedLine(f, (c[0],c[1]), (int(c[0]+dx), int(c[1]+dy)),int(norm), 1)
                    if r is not None:
                        print('Ball detected',c, r)
                        f=cv.circle(f, (c[0], c[1]), r, (0,0,0), 2)
                cv.imwrite('color.jpg', f)
            tmp_c = c        
            #else:
                #tmp_c = c
        search_on = True
        ball_moving=True
        shooting_condition=True

            



    #phase 2 get images and decide


def main(IP,PORT):
    imgs = load_imgs(1,300)
    #print(imgs)
    imgs_to_movie = []
    tmp_c = None
    tmp_r = None
    largest_vector = None

    #opencv saves as (y,x)
    for i,img in enumerate(imgs):
        img = cv.resize(img, (800,600))
        #import ipdb;ipdb.set_trace()
        f,c,r = detect_keypoint_yolo(img)
        f = cv.resize(f, (800,600))
        angle=9000
        angle_='N'
        #print(c)
        if tmp_c is None and c is not None:
            tmp_c = (c[0], c[1])
        elif tmp_c is not None and c is not None:
            out_pixel, angle_, dx, dy, norm =convert_centers2out(c, tmp_c)
            angle_ = output_pixel2decision(out_pixel, r)
            print(angle_)
            f = cv.arrowedLine(f, (c[0],c[1]), (int(c[0]+dy), int(c[1]+dx)),int(r), 2)
            f=cv.circle(f, (c[0]*10, c[1]*10), r, (0,0,0), 2)
            
        tmp_c = c

        if r is not None:
            #print('Ball detected',c, r)
            f=cv.circle(f, (c[0]*10, c[1]*10), r*10, (0,0,0), 2)
        #pil_im = Image.fromarray(f)
        #draw = ImageDraw.Draw(pil_im)
        #font = ImageFont.truetype("arial.ttf", 30)
        #f = np.array(pil_im)

        #draw.text((0, 0), angle_, font=font)
        cv.imshow('w', f)
        cv.waitKey(0)
        #cv.imwrite('tomovie/'+str(i).zfill(3)+'.jpg',f)
    (
    ffmpeg
    .input('tomovie/*.jpg', pattern_type='glob', framerate=10)
    .output('frames.mp4')
    .run()
    )

def main_(IP, PORT):
    """
    First get an image, then show it on the screen with PIL.
    """
    # Get the service ALVideoDevice.
    motion = ALProxy("ALMotion", IP, PORT)
    video_service = ALProxy("ALVideoDevice",'10.0.7.101', 9559)
    tracker = ALProxy("ALTracker", '10.0.7.101', 9559) 
    motion = ALProxy("ALMotion", IP, PORT)
    posture = ALProxy("ALRobotPosture", IP, PORT)
    resolution = 7    # VGA
    colorSpace = 11   # RGB
    motion.wakeUp()

    fractionMaxSpeed = 0.8
    # Go to posture stand
    posture.goToPosture("StandInit", fractionMaxSpeed)
    videoClient = video_service.subscribe("python_client", resolution, colorSpace, 30)
    
    
    targetName = "RedBall"
    diameterOfBall = 0.06
    tracker.registerTarget(targetName, diameterOfBall)
    

    # set mode
    mode = "Head"
    tracker.setMode(mode)

    # Then, start tracker.
    tracker.track(targetName)
    t0 = time.time()
    counter =0
    # Get a camera image.
    # image[6] contains the image data passed as an array of ASCII chars.
    while 1:
        naoImage = video_service.getImageRemote(videoClient)

        counter +=1
        # Now we work with the image returned and save it as a PNG  using ImageDraw
        # package.

        # Get the image size and pixel array.
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]
        image_string = str(bytearray(array))
        print('img dimensions',imageHeight, imageWidth)
        # Create a PIL Image from our pixel array.
        im = np.fromstring(image_string, np.uint8).reshape([imageHeight, imageWidth, 3])
        t1 = time.time()
        detect_keypoint(im, counter)
        print('Time BAll detection',time.time()-t1)
        # Save the image.
        print(im.shape)
        print('Time Image retrieval',time.time()-t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.0.7.101",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
  
    IP=args.ip
    PORT = args.port
    
    posture = ALProxy("ALRobotPosture", IP, PORT)
    motion = ALProxy("ALMotion", IP, PORT)
    video_service = ALProxy("ALVideoDevice",IP, PORT)
    tracker = ALProxy("ALTracker", IP, PORT) 
    tts = ALProxy("ALTextToSpeech", IP, PORT)
    tts.say('connected')
    resolution = 7    # VGA
    colorSpace = 11   # RGB
    motion.wakeUp()

    fractionMaxSpeed = 0.2
    # Go to posture stand
    posture.goToPosture("Crouch", fractionMaxSpeed)
    videoClient = video_service.subscribe("python_client", resolution, colorSpace, 30)
    
    
    # Add target to track.
    ballSize=0.06
    targetName = "RedBall"
    diameterOfBall = ballSize
    tracker.registerTarget(targetName, diameterOfBall)

    # set mode
    mode = "Head"
    tracker.setMode(mode)
    names = ['HeadYaw', 'HeadPitch']
    angles = [0.0, 0.1]
    
    # Then, start tracker.
    tracker.track(targetName)

    t0 = time.time()
    counter =0
    ball_moving = True
    error_margin = 0.1
    counter = 0
    non_moving_frames = 5
    tmp_coordinates = tracker.getTargetPosition()
    #first phase find the ball
    #tts.say('toggleSearch')
    #search_on = True
    #tracker.toggleSearch(True)
    
    while True:
        #tracker.track(targetName)
        t0 = time.time()
        counter =0
        ball_moving = True
        error_margin = 0.1
        counter = 0
        non_moving_frames = 15
        tmp_coordinates = tracker.getTargetPosition()
        #first phase find the ball
        tts.say('toggleSearch')
        search_on = True
        #tracker.toggleSearch(True)
        ball_tracke_counter =0
        while not tracker.isTargetLost() and  ball_tracke_counter<200:
            print(tracker.isTargetLost())
            print('still in search loop')
            print(tracker.getTargetPosition())
            #tracker.toggleSearch(True)
            #cv.imwrite('color.jpg', f)
            ball_tracke_counter += 1
            print(ball_tracke_counter)
        tracker.stopTracker()
        tts.say('I located the ball')
        tts.say('Expecting a shooting attempt')
        motion.setAngles(names, angles, fractionMaxSpeed)

        shooting_condition= True
        tmp_c = None
        angle=9000
        angle_='N'
        movement_threshold = 20
        visualize = True
        while shooting_condition:

            naoImage = video_service.getImageRemote(videoClient)
            
            counter +=1

            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_string = str(bytearray(array))
            img = np.fromstring(image_string, np.uint8).reshape([imageHeight, imageWidth, 3])
            img = cv.resize(img, (800,600))
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            f,c,r = detect_keypoint_yolo(img)
            #print(c,r)
            
            if tmp_c is None:
                tmp_c = c
            elif tmp_c is not None and c is not None:
    
                out_pixel, angle, dx, dy, norm =convert_centers2out(c, tmp_c)
                #print('out,angle, dx,dy,norm',out_pixel, angle, dx, dy, norm )
                decision = output_pixel2decision(out_pixel, r)
                print('decision/norm/c: ',decision, norm, c)
                if norm > movement_threshold  and decision!='------':  
                    #shooting_condition = False
                    tts.say(decision)
                    #break

                if visualize:
                    #f = cv.resize(f, (800,600))
                    f = cv.arrowedLine(f, (c[0],c[1]), (int(c[0]+dy), int(c[1]+dx)),int(norm)*20, 1)
                    if r is not None and r < 8:
                        #print('Ball detected',c, r)
                        f=cv.circle(f, (c[0], c[1]), r, (0,0,0), 2)
            cv.imwrite('color.jpg', f)
            tmp_c = c        
            #else:
                #tmp_c = c
        search_on = True
        ball_moving=True
        shooting_condition=True
