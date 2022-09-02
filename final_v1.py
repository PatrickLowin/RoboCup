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
import math
import zmq
import json, ast
from utils import *
context = zmq.Context()
import time


if __name__ == "__main__":
    #  Socket to talk to server
    print("Connecting to hello world server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="10.0.7.101",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
  
    IP=args.ip
    PORT = args.port
    

    #Robot Proxies for Communication
    posture = ALProxy("ALRobotPosture", IP, PORT)
    motion = ALProxy("ALMotion", IP, PORT)
    video_service = ALProxy("ALVideoDevice",IP, PORT)
    tracker = ALProxy("ALTracker", IP, PORT) 
    tts = ALProxy("ALTextToSpeech", IP, PORT)
    tts.say('connected')
    resolution = 7    # VGA
    colorSpace = 11   # RGB

    #head turning
    kopfDrehenRechts = True
    #degree of head turns
    ANGLE_HEAD_DEG = 25

    #motion.wakeUp()

    fractionMaxSpeed = 0.2
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
    names = ['HeadPitch']
    angles = [0.3941]
    
    # Then, start tracker. Kind of useless since we only use the CNN for detections
    tracker.track(targetName)

    t0 = time.time()
    counter =0
    ball_moving = True
    error_margin = 0.1
    counter = 0
    countTracks = 5
    tmp_coordinates = tracker.getTargetPosition()

    motion.setAngles(names, angles, fractionMaxSpeed)

    while True:
        #tracker.track(targetName)
        t0 = time.time()
        counter =0
        ball_moving = True
        error_margin = 0.1
        counter = 0
        countTracks = 10
        tmp_coordinates = tracker.getTargetPosition()
        #first phase find the ball
        ball_tracke_counter =0
        isTargetLost = tracker.isTargetLost()
        while isTargetLost:
            #get image and detect ball
            naoImage = video_service.getImageRemote(videoClient)
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_string = str(bytearray(array))
            img = np.fromstring(image_string, np.uint8).reshape([imageHeight, imageWidth, 3])
            img = cv.resize(img, (800,600))
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            f,c,r = detect_keypoint_yolo(img)
            #write image for visualization
            cv.imwrite('color.jpg', img)
            print(c)
            #print(motion.getAngles("HeadPitch", False))
            if c is not None:
                if c[0]>150 and c[0]<600:
                    isTargetLost=False
                    break
            
            #head turning
            print "Target lost"
            counter_l = 0
            counter_r = 0
            counter_m = 0
            xValues = []
            yValues = []
            counter_desicions = 0
            angle = motion.getAngles("HeadYaw", False)
            print angle 
            print np.degrees(angle)  
            print math.radians(ANGLE_HEAD_DEG)
            test = math.radians(ANGLE_HEAD_DEG)
            print np.degrees(test)
            if angle[0] <= math.radians(ANGLE_HEAD_DEG * (-1)):
                print "ist jetzt FALSE"
                kopfDrehenRechts = False
            if angle[0] > math.radians(ANGLE_HEAD_DEG):
                print "ist jetzt TRUE"
                kopfDrehenRechts = True
            

            if kopfDrehenRechts:
                newAngle = angle[0] - math.radians(20)
                motion.angleInterpolation(["HeadYaw"],
                [newAngle],
                [0.5],
                True)
            else:
                newAngle = angle[0] + math.radians(20)
                motion.angleInterpolation(["HeadYaw"],
                [newAngle],
                [0.5],
                True)


        #wait x Detections 
        while ball_tracke_counter < countTracks:
            naoImage = video_service.getImageRemote(videoClient)

            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_string = str(bytearray(array))
            img = np.fromstring(image_string, np.uint8).reshape([imageHeight, imageWidth, 3])
            img = cv.resize(img, (800,600))
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            f,c,r = detect_keypoint_yolo(img)
            if c is not None:
                ball_tracke_counter+= 1
            else:
                shooting_condition=False

        tracker.stopTracker()
        tts.say('I located the ball')
        tts.say('Expecting a shooting attempt')
        motion.setAngles(names, angles, fractionMaxSpeed)
        
        #if all images contain the ball we expect a shot
        if ball_tracke_counter>=countTracks:
            shooting_condition= True

        tmp_c = None
        angle=9000
        angle_='N'
        movement_threshold = 30
        visualize = True

        while shooting_condition:
            #get images from NAO and detect ball
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
            print(c)
            #print(motion.getAngles("HeadPitch", False))
            if tmp_c is None:
                tmp_c = c
            elif tmp_c is not None and c is not None:
                #Compute Trajectory
                out_pixel, angle, dx, dy, norm =convert_centers2out(c, tmp_c)
                #print('out,angle, dx,dy,norm',out_pixel, angle, dx, dy, norm )
                decision = output_pixel2decision(out_pixel, r)
                print('decision/norm/c: ',decision, norm, c)
                if norm > movement_threshold  and decision!='------':  
                    shooting_condition = False
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
        tracker.track(targetName)
        search_on = True
        ball_moving=True
        shooting_condition=True
