import os
import pandas as pd
import torch 
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import torchvision.transforms as T
from PIL import Image
class BallDataset(Dataset):
    def __init__(self, real_path='datasets/annotations', synthetic_path='datasets/synthetic', transform=None, target_transform=None):
        self.data_real_path = real_path
        self.data_syn_path = synthetic_path
        self.bg_files = os.listdir(os.path.join(synthetic_path,'bg_'))
        self.ball_files = os.listdir(os.path.join(synthetic_path,'balls'))

        self.real_img =[]
        self.real_label=[]
        self.load_real_data()

        self.len_bg = len(self.bg_files)
        self.len_balls = len(self.ball_files)
        self.len_real = len(self.real_img)

        transforms = torch.nn.Sequential(
            T.ColorJitter(brightness=.7,contrast=0.6, saturation=0.5),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.Resize(size=(60,80)),
            T.Resize(size=(600,800)))
            #T.RandomRotation(degrees=(0, 180)),
            #T.RandomPerspective(distortion_scale=0.6, p=1.0)
        real_transforms = torch.nn.Sequential(
            T.ColorJitter(brightness=.3,contrast=0.4, saturation=0.5),
            T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
            T.Resize(size=(60,80)),
            T.Resize(size=(600,800)))
        self.syn_transform = transforms
        self.real_transform = real_transforms
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)



    def load_real_data(self):
        imgs_filenames = sorted(os.listdir(os.path.join(self.data_real_path, 'imgs')))
        kp_filenames = sorted(os.listdir(os.path.join(self.data_real_path, 'kp')))

        for img, kp in zip(imgs_filenames,kp_filenames):
            img_file_full_path = os.path.join(self.data_real_path, 'imgs', img)
            kp_file_full_path = os.path.join(self.data_real_path, 'kp', kp)
            self.real_img.append(cv.imread(img_file_full_path))
            self.real_label.append(np.genfromtxt(kp_file_full_path, delimiter=','))


    def get_synthetic_img(self):
        idx_bg = np.random.randint(0, self.len_bg)
        idx_ball = np.random.randint(0, self.len_balls)

        bg_path = os.path.join(self.data_syn_path,'bg_', self.bg_files[idx_bg])
        ball_path = os.path.join(self.data_syn_path,'balls', self.ball_files[idx_ball])
        
        ball = cv.imread(ball_path)
        bg= cv.imread(bg_path)
        if np.random.rand()>0.5:
            bg = cv.flip(bg,0)

        if np.random.rand()>0.5:
            bg = cv.rotate(bg,cv.ROTATE_180)

        random_scalor = 1/np.random.randint(1,3) 
        ball = cv.resize(ball,(int(random_scalor*160), int(random_scalor*120)))
        bg = cv.resize(bg, (800,600))
        random_y = np.random.randint(0,600 - ball.shape[0])
        random_x = np.random.randint(0,800 - ball.shape[1])

        bg_crop = bg[random_y:random_y+ball.shape[0], random_x: random_x+ball.shape[1]]
        ball_gray = cv.cvtColor(ball, cv.COLOR_BGR2GRAY)
        
        if ball_gray[0,0]==0:
            ball_gray[ball_gray<15]=255
        
        ball_mask = np.zeros_like(ball_gray)
        ball_mask[ball_gray>250]=1
        ball_mask = cv.morphologyEx(ball_mask, cv.MORPH_OPEN, np.ones((3,3),np.uint8))
        cv.imshow('ball', ball_gray)
        cv.imshow('ballmask', ball_mask*255)
        ball_on_bg = bg_crop
        print(ball_on_bg.shape, ball.shape, ball_mask.shape)
        try:
            ball_on_bg[ball_mask==0]=ball[ball_mask==0]
            bg[random_y:random_y+ball.shape[0], random_x: random_x+ball.shape[1]]=ball_on_bg
        except:
            import ipdb;ipdb.set_trace()
        label = [[random_y, random_x], [random_y+ball.shape[0],random_x+ball.shape[1]]]
        return bg, np.asarray(label)

    def get_real_img(self, idx):
        return self.real_img[idx], self.real_label[idx]


    def convert(im, box):
        w= int(im.size[0])
        h= int(im.size[1])
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def label2yolo(self,label, W=800, H=600):
        cv_xy1 = np.flip(label[0])
        cv_xy2 = np.flip(label[1])
        x_mean = (cv_xy1[0] + cv_xy2[0])/2.0
        y_mean = (cv_xy1[1] + cv_xy2[1])/2.0
        b_w = (cv_xy2[0] - cv_xy1[0])/W
        b_h = (cv_xy2[1] - cv_xy1[1])/H
        x_mean = x_mean/W
        y_mean = y_mean/H

        return [0, x_mean,y_mean,b_w,b_h]



    def __getitem__(self, idx):
        rand = np.random.rand()

        if idx < self.len_real*1.5:
            print(idx)
            
            img = self.real_img[idx%self.len_real]
            label = self.real_label[idx%self.len_real]

            if self.real_transform:
                im = Image.fromarray(np.uint8(img))
                im = self.real_transform(im)
                img = np.array(im)
        else:
            img, label = self.get_synthetic_img()
            if self.syn_transform:
                im = Image.fromarray(np.uint8(img))
                im = self.syn_transform(im)
                img = np.array(im)

        

        #if self.target_transform:
            #label = self.target_transform(label)

        if False:
            try:
                if label.shape==(2,2):
                    cv.rectangle(img, np.flip(label[0]).astype(int), 
                                    np.flip(label[1]).astype(int), (0,0,255), 2)
            except:
                import ipdb;ipdb.set_trace()
        
        #cv.imshow('image', img)
        #cv.waitKey(0)
        if len(label)==2:
            try:
                label = self.label2yolo(label, 800, 600)
            except:
                cv.imshow('image', img)
                cv.waitKey(0)
                label=[]

        return img, label


Ball = BallDataset()
for i in range(0,5000):
    img, label = Ball.__getitem__(i)
    if len(label)!=0:
        cv.imwrite('datasets/coco128/images/'+str(i).zfill(5)+'.jpg', img)
        with open("datasets/coco128/labels/"+str(i).zfill(5)+".txt", "w") as text_file:
            text_file.write(str(label[0])+' '+str(label[1])+' '+str(label[2])+' '+str(label[3])+' '+str(label[4]))
    else:
        one=1
        #cv.imwrite('synthetic/bg_'+str(i).zfill(5)+'.jpg', img)
