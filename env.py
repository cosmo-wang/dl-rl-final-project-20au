import sys
import json
import torch
import torchvision
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
from src.DDPG import Painter
from src.util import *
from PIL import Image
from torchvision import transforms, utils
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aug = transforms.Compose(
            # [transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # ONLY FOR MNIST !!!!
            [transforms.ToPILImage(),
            transforms.Resize(128), # ONLY FOR MNIST !!!!
            transforms.RandomHorizontalFlip(),
             ])

width = 128 
convas_area = width * width

# img_train = []
# img_test = []
# train_num = 0
# test_num = 0

class Paint:
    def __init__(self, batch_size, max_step):
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = (13)
        # self.observation_space = (self.batch_size, width, width, 7)  # for RGB
        self.observation_space = (self.batch_size, width, width, 3)  # for grayscale
        self.test = False

        self.img_train = []
        self.img_test = []
        self.train_num = 0
        self.test_num = 0

        self.painter = Painter('./renderer.pkl')
        self.painter.to(device)
        
    def load_data(self, dataset):
        """
        @param dataset: A String representing the dataset
        """
        if dataset == "MNIST":
            self.img_train = datasets.MNIST(root='./data', train=True, download=True, transform=None).data
            self.img_test = datasets.MNIST(root='./data', train=False, download=True, transform=None).data
            self.train_num = len(self.img_train)
            self.test_num = len(self.img_test)
        else: # CelebA
            for i in range(200000):
                img_id = '%06d' % (i + 1)
                try:
                    img = cv2.imread('./data/img_align_celeba/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img, (width, width))
                    if i > 2000:                
                        self.train_num += 1
                        self.img_train.append(img)
                    else:
                        self.test_num += 1
                        self.img_test.append(img)
                finally:
                    if (i + 1) % 10000 == 0:                    
                        print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))
        
    def pre_data(self, id, test):
        if test:
            img = self.img_test[id]
        else:
            img = self.img_train[id]
        # For CelebA:
        # if not test:
        #     img = aug(img)
        # For MNIST:
        img = aug(img)
        img = np.asarray(img)
        return img  # for grayscale
        # return np.transpose(img, (2, 0, 1))  # for RGB
    
    def reset(self, test=False, begin_num=False):
        self.test = test
        self.imgid = [0] * self.batch_size
        # self.gt = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)  # for RGB
        self.gt = torch.zeros([self.batch_size, 1, width, width], dtype=torch.uint8).to(device)  # for grayscale
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num)  % self.test_num
            else:
                id = np.random.randint(self.train_num)
            self.imgid[i] = id
            self.gt[i] = torch.tensor(self.pre_data(id, test))
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        # self.canvas = torch.zeros([self.batch_size, 3, width, width], dtype=torch.uint8).to(device)  # for RGB
        self.canvas = torch.zeros([self.batch_size, 1, width, width], dtype=torch.uint8).to(device)  # for grayscale
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()
    
    def observation(self):
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # T B * 1 * width * width
        ob = []
        T = torch.ones([self.batch_size, 1, width, width], dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.gt, T.to(device)), 1) # canvas, img, T

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    def step(self, action):
        self.canvas = (self.painter.paint(action, self.canvas.float() / 255)[0] * 255).byte()
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)
    
    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)