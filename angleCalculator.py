import math

NUMBER_OF_DECIMALS = 3
THRESHOLD_SHOULD_FALL = 1
THRESHOLD_TOO_FAR_AWAY = 5
THRESHOLD_SPEED = 3 # 3 Times

t1 = [1.0873615741729736, -0.3478677570819855, -0.1748875230550766]

# wait some seconds

t2 = [0.7486830353736877, -0.1086355671286583, -0.1625957489013672]


def roundArray(n):
    return round(n, NUMBER_OF_DECIMALS)

t1 = map(roundArray, t1)
t2 = map(roundArray, t2)


def calcWinkel2D(u,v):
    x1 = float(u[0])
    y1 = float(u[1])
    x2 = float(v[0])
    y2 = float(v[1])


    xDiff = float(x2-x1)
    yDiff = float(y2-y1)

    if(xDiff == 0 and yDiff == 0):
        print "Ball bewegt sich nicht"
        return "NONE"
    if(yDiff == 0):
        print "Ball rollt nicht auf das Tor zu"
        return "NONE"

    expectedDistanceToRobotOnBaseline = None

    if x2-x1 == 0:
        if y2 > y1:
            print "Ball rollt (gerade) vom Ziel weg"
            return "NONE"

        print "Ball rollt gerade"
        expectedDistanceToRobotOnBaseline = x2   
    else:
        if y2 > y1:
            print "Ball rollt vom Ziel weg"
            return "NONE"
        m = yDiff / xDiff 
        n = y1 - (m * x1)
        expectedDistanceToRobotOnBaseline = (n * (-1)) / m

    # Test Entfernungs-Checker
    d = math.sqrt(xDiff**2 + yDiff**2) #Entfernung zwischen beiden Punkten
    q = math.sqrt((x2-expectedDistanceToRobotOnBaseline)**2 + y2**2)
    speedIndicator = q / d
    print "speedIndicator:", speedIndicator
    if(speedIndicator > THRESHOLD_SPEED):
        print "SLOW BALL"    
    # Test Ende

    print "Distance from Robot on Baseline: ", expectedDistanceToRobotOnBaseline
    if abs(expectedDistanceToRobotOnBaseline) < THRESHOLD_SHOULD_FALL:
        return "CENTER"
    if abs(expectedDistanceToRobotOnBaseline) > THRESHOLD_TOO_FAR_AWAY:
        return "NONE"
    if THRESHOLD_SHOULD_FALL < expectedDistanceToRobotOnBaseline <THRESHOLD_TOO_FAR_AWAY:
        #positive values to the left
        return "LEFT"
    if THRESHOLD_SHOULD_FALL * (-1) > expectedDistanceToRobotOnBaseline > THRESHOLD_TOO_FAR_AWAY * (-1):
        return "RIGHT"


a = [-4, 6]
b = [-1, 3]
s = calcWinkel2D(a,b)
print s
 
