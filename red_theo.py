# -*- encoding: UTF-8 -*-

"""
This example shows how to use ALTracker with red ball.
"""

import time
import argparse
from naoqi import ALProxy
from angleCalculator import calcWinkel2D
#import cv2 as cv
import numpy as np
from scipy import stats
import math 
'''
counter_m = 0
counter_r = 0
counter_l = 0
counter_decisions = 0
xValues = []
yValues = []
'''
def main(IP, PORT, ballSize):
    counter = 0
    tts = ALProxy("ALTextToSpeech", IP, PORT)
    tts.say('connected')
    print "Connecting to", IP, "with port", PORT
    motion = ALProxy("ALMotion", IP, PORT)
    posture = ALProxy("ALRobotPosture", IP, PORT)
    tracker = ALProxy("ALTracker", IP, PORT)
    videoRec = ALProxy("ALVideoRecorder", IP,PORT)
    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    resolution = 2    # VGA
    colorSpace = 11   # RGB

    tts = ALProxy("ALTextToSpeech", IP, 9559)
    tts.say('proxies set up')
    # First, wake up.
    #motion.wakeUp()

    fractionMaxSpeed = 0.8
    # Go to posture stand
    posture.goToPosture("Crouch", fractionMaxSpeed)

    # Add target to track.
    targetName = "RedBall"
    diameterOfBall = ballSize
    tracker.registerTarget(targetName, diameterOfBall)

    # set mode
    mode = "Head"
    tracker.setMode(mode)

    # Then, start tracker.
    tracker.track(targetName)

    print "ALTracker successfully started, now show a red ball to robot!"
    print "Use Ctrl+c to stop this script."
    

    SLEEP_TIME_IN_SEC = 0.05
    # BUFFER_KOEFFIZENT = 2
    # MAX_LENGTH_OF_BUFFER = int((1 / SLEEP_TIME_IN_SEC) * BUFFER_KOEFFIZENT) #feste Groesse?
    # print "MAX_LENGTH_OF_BUFFER", MAX_LENGTH_OF_BUFFER
    MAX_LENGTH_OF_BUFFER = 20 #feste Groesse ja

    MESSFEHLER = 0.4
    THRESHOLD_MIDDLE = 0.25
    THRESHOLD_OUTSIDE = 0.7

    xValues = []
    yValues = []
    
    counter_m = 0
    counter_r = 0
    counter_l = 0
    counter_decisions = 0
    
    
    xMin =  1000
    xMax = -1000
    yMin =  1000
    yMax = -1000

    kopfDrehenRechts = True
    ANGLE_HEAD_DEG = 45
    '''
    def resetAll():
        resetCounter()
        global xValues, yValues
        xValues = []
        yValues = []

    def resetCounter():
        global counter_l, counter_r, counter_m, counter_decisions
        print counter_l, counter_r, counter_m, counter_decisions
        counter_l = 0
        counter_r = 0
        counter_m = 0
        counter_decisions = 0
    
    def increaseCounter(counter):
        global counter_decisions
        print "counter increased"
        # counter += (desicions + 2)
        counter_desicions += 1
        return desicions + 1
    '''

    try:
        posture.goToPosture("Crouch", fractionMaxSpeed)
        while True:
            time.sleep(SLEEP_TIME_IN_SEC)

            isTargetLost = tracker.isTargetLost()
            if(isTargetLost == True):
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
                continue

            pos = tracker.getTargetPosition()
	        
            #pos ist leer
            if(len(pos) == 0):
		        continue 

            # Ball ist zu hoch
            if(pos[2] > -0.3):
                continue

            xValues.append(pos[0])
            yValues.append(pos[1])
            
            if len(xValues) > MAX_LENGTH_OF_BUFFER:
                xValues.pop(0)
                yValues.pop(0)

            xMin = min(xMin, pos[0])
            xMax = max(xMax, pos[0])
            yMin = min(yMin, pos[1])
            yMax = max(yMax, pos[1])

            xDelta = abs(xValues[0] - xValues[-1])
            yDelta = abs(yValues[0] - yValues[-1])

            if(xDelta < MESSFEHLER and yDelta < MESSFEHLER):
                print "Keine signifikante Änderung im Buffer"
                counter_l = 0
                counter_r = 0
                counter_m = 0
                counter_desicions = 0
                continue
            elif(len(xValues) <= (MAX_LENGTH_OF_BUFFER / 2)):
                #print "No Action"
                continue
                    
            xStartDurchschnitt = (xValues[0] + xValues[1] + xValues[2]) / 3
            xEndDurchschnitt = (xValues[-1] + xValues[-2] + xValues[-3]) / 3
            
            if(xStartDurchschnitt - xEndDurchschnitt < 0):
                print "Falsche Richtung"
                counter_l = 0
                counter_r = 0
                counter_m = 0
                counter_desicions = 0
                continue        
            
            slope, intercept, r, p, std_err = stats.linregress(xValues, yValues)
            print "intercept:", intercept

            if(intercept > THRESHOLD_MIDDLE):
                #counter_l = increaseCounter(counter_l)
                counter_l += (counter_decisions + 2)
                counter_desicions += 1
                print "L"
            elif(intercept < (THRESHOLD_MIDDLE * (-1))):
                #counter_r = increaseCounter(counter_r)
                counter_r += (counter_decisions + 2)
                counter_desicions += 1
                print "R"
            elif((-1)* THRESHOLD_MIDDLE < intercept < THRESHOLD_MIDDLE):
    	        #counter_m = increaseCounter(counter_m)
                counter_m += (counter_decisions + 2)
                counter_desicions += 1
    	        print "M"
            else:
                print "NONE"
		
            print counter_l, counter_m, counter_r, counter_decisions


            ANZAHL = 10 #ersten 4 Entscheidungen oder 5. und 6. Entscheidung , davor 10
            if(counter_l >= ANZAHL and counter_l > counter_m and counter_l > counter_r):
                tts.say("Left")
                counter_l = 0
                counter_r = 0
                counter_m = 0
                xValues = []
                yValues = []
                counter_desicions = 0
            elif(counter_r >= ANZAHL and counter_r > counter_m and counter_r > counter_l):
                print "final Desicion: RIGHT" 
                tts.say("Right")
                counter_l = 0
                counter_r = 0
                counter_m = 0
                xValues = []
                yValues = []
                counter_desicions = 0
            elif(counter_m >= ANZAHL):
                print "final Desicion: MIDDLE" 
                tts.say("Middle")
                counter_l = 0
                counter_r = 0
                counter_m = 0
                xValues = []
                yValues = []
                counter_desicions = 0
		            
    except KeyboardInterrupt:
        print
        print "Interrupted by user"
        print "Stopping..."

    tracker.stopTracker()
    tracker.unregisterAllTargets()
    motion.rest()

    print "ALTracker stopped."


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="nao6.local",
                        help="Robot ip address.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number.")
    parser.add_argument("--ballsize", type=float, default=0.06,
                        help="Diameter of ball.")

    args = parser.parse_args()

    main(args.ip, args.port, args.ballsize)


