import cv2
import torch
from PIL import Image
import os 
import numpy as np
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# Model
def test():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp17/weights/best.pt', device='cpu')
    model.cpu()
    # Images
    # download 2 images
    #im1 = Image.open('../test3.jpg')  # PIL image
    im2 = cv2.imread('../frame5.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
    im3 = cv2.imread('../test4.png')[..., ::-1]  # OpenCV image (BGR to RGB)
    imgs = [im2]  # batch of images
    results = model(imgs, size=128)  # includes NMS
    results.print()  
    results.show()
    # Inference

    vidcap = cv2.VideoCapture('../redball_test.mp4')
    success,image = vidcap.read()
    count = 0
    img_list = []
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 

    success,image = vidcap.read()
    video = cv2.VideoWriter('test_video.mp4', fourcc, 24, (image.shape[1],image.shape[0]))

    while success:
        #cv.imwrite("datasets/rgb1/frame%d.jpg" % count, image)     # save frame as JPEG file      

        print('Read a new frame: ', success)
        
        count += 1

        # Results
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_list.append(pil_img)
        results = model(img_list, size=128)  # includes NMS
        #import ipdb;ipdb.set_trace()
        video.write(np.array(results.render())[0][:,:,::-1])
        success,image = vidcap.read()
        #results.print()  
        #results.show()  # or .show()
        img_list = []
        #cv2.imshow('sanity', image)
        #cv2.waitKey(0)

    cv2.destroyAllWindows()
    video.release()


import time
import zmq
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
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


def get_message_and_send_prediciton():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp17/weights/best.pt', device='cpu')
    model.cpu()


    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        #  Wait for next request from client
        img_message = recv_array(socket)
        results = model([img_message[:,:,::-1]], size=128)
        #results.save()
        print("Received request")
        #import ipdb;ipdb.set_trace()
        #  Do some 'work'
        #convert bbox to center + ~radius
        if len(results.xyxyn[0])>0:
            xyxyn= results.xyxyn[0][0]
            xyxyn[5]=min(xyxyn[2]-xyxyn[0], xyxyn[3]-xyxyn[1])/2
            centerx = (xyxyn[0]+xyxyn[2])/2
            centery = (xyxyn[1]+xyxyn[3])/2
            A = np.array([centerx, centery, xyxyn[4], xyxyn[5]], dtype=np.float32)
        else:
            A = np.array([-100, -100, -100, -100],dtype=np.float32)
        #convert to 800x600 frame
        #maybe switch 600 and 800 depending on opencv
        # LAST ENTRY IS RADIUS


        #  Send reply back to client
        send_array(socket,A)
        
get_message_and_send_prediciton()