import torchvision
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
from IPython.display import display
import ipywidgets
import traitlets
from jetbot import Camera, bgr8_to_jpeg
from jetbot import Robot

device = torch.device('cuda')

ca_model = torchvision.models.resnet18(pretrained=False)
ca_model.fc = torch.nn.Linear(512, 2)
ca_model.load_state_dict(torch.load('best_model_resnet18.pth'))

ca_model = ca_model.to(device)
ca_model = ca_model.eval().half()

steering_model = torchvision.models.resnet18(pretrained=False)
steering_model.fc = torch.nn.Linear(512, 2)
steering_model.load_state_dict(torch.load('best_steering_model_xy.pth'))

steering_model = steering_model.to(device)
steering_model = steering_model.eval().half()


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
normalize = torchvision.transforms.Normalize(mean, std)

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


camera = Camera()

robot = Robot()

speed_gain: float = 0.290
steering_gain: float = 0.030
steering_dgain: float = 0.015
steering_bias: float = -0.010

angle: float = 0.0
angle_last: float = 0.0


def road_following(change):
    global angle, angle_last

    image = change['new']
    xy = steering_model(preprocess(image)).detach().float().cpu().numpy().flatten()
    
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

    angle = np.arctan2(x, y)
    pid = angle * steering_gain + \
        (angle - angle_last) * steering_dgain
    angle_last = angle

    steering_value: float = pid+steering_bias

    # print("x = {x},y = {y} ,speed = {speed},steering= {steering}".format(
    #     x=x, y=y, speed=speed, steering=steering_value))

    robot.left_motor.value = max(
        min(speed_gain + steering_value, 1.0), 0.0)
    robot.right_motor.value = max(
        min(speed_gain - steering_value, 1.0), 0.0)


def execute(change):
    x = change['new']
    x = preprocess(x)
    y = ca_model(x)
    
    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim=1)
    
    prob_blocked = float(y.flatten()[0])

    if prob_blocked<0.5:
        road_following(change)
    else:
        robot.stop()
  

execute({'new': camera.value})

camera.observe(execute, names='value')
