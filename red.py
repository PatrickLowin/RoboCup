# -*- encoding: UTF-8 -*-

"""
This example shows how to use ALTracker with red ball.
"""

import time
import argparse
from naoqi import ALProxy
from angleCalculator import calcWinkel2D
import cv2 as cv
import numpy as np
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
    motion.wakeUp()

    fractionMaxSpeed = 0.8
    # Go to posture stand
    posture.goToPosture("StandInit", fractionMaxSpeed)

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
    # # videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 15)
    # videoRec.setFrameRate(10.0)
    # videoRec.setResolution(2)
    # videoRec.startRecording('/home/nao/recordings/cameras','video')

    try:

        print tracker.getTargetPosition()
        lastValue = tracker.getTargetPosition()
        while True:
            time.sleep(0.1)
            newValue = tracker.getTargetPosition()
            print newValue
            #print(lastValue)
            #print(newValue)
            # if not len(lastValue)==0 and not len(newValue)==0:
            #     print calcWinkel2D(lastValue, newValue)
            # lastValue = newValue

            if False:

                naoImage = camProxy.getImageRemote(videoClient)
                #camProxy.unsubscribe(videoClient)

                imageWidth = naoImage[0]
                imageHeight = naoImage[1]
                array = naoImage[6]
                image_string = str(bytearray(array))

                # Create a PIL Image from our pixel array.
                img = np.fromstring(image_string, np.uint8).reshape([imageHeight, imageWidth, 3])
                #cv.imwrite('./imgs/IMG'+str(counter).zfill(6)+'.png', img)
                #np.save('./position/pos'+str(counter).zfill(6), np.asarray(tracker.getTargetPosition()))
                counter= counter+1
        # videoInfo = videoRec.stopRecording()
        print(videoInfo[1])
    except KeyboardInterrupt:
        print
        print "Interrupted by user"
        print "Stopping..."
    # videoInfo = videoRec.stopRecording()
    # Stop tracker, go to posture Sit.
    tracker.stopTracker()
    tracker.unregisterAllTargets()
    # posture.goToPosture("Sit", fractionMaxSpeed)
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