How to start:
1. Install the requirements for YOLO in a new environment with Python 3.9
2. Install the requirements for NAO in a new environment with Python 2.7
	-export PYTHONPATH=${PYTHONPATH}:/path/to/pynaoqi
	
Start CNN Server:
    In /code-for-nao/YOLO_train/yolov5 run python inference.py
Then:
    In /code-for-nao run python final_v1.py

This creates a CNN server that evaluates images and connects to NAO to perform goalkeeping activities

In YOLO_train we also have the annotation.py to annotate images.
    specify your working directory (ie where the images are) and then click and drag an (invisible) rectangle around the ball

With render_balls we create a dataset of real and synthetic images.
We also include a function to transform bbox coordinates to yolo format

###
The dataset was to big to upload it on ISIS. I will just create a git

###

/datasets contains the labeld images in the /datasets/coco folder (in yolo format) and when creating a dataset with annotations.py we also need to convert the data to yolo format. A function is provided in the class.


-frames.mp4 shows an example of the trajectory prediction. However Nao head moved (which we later turned of) so the predictions are not quite accurate.

-test_video.mp4 shows an example of the YOLOv5 trained on our synthtic and real images from the lab. We wanted to test how well it works on unseen data and therefore choose a video where the floor is blue instead of green to make it harder.


