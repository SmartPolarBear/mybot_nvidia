from IPython.display import display
from jetbot import Robot
from jetbot import Camera, bgr8_to_jpeg
import traitlets
import ipywidgets
import numpy as np
import PIL.Image
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch2trt import TRTModule
import torch

device = torch.device('cuda')

steering_model_trt = TRTModule()
steering_model_trt.load_state_dict(torch.load('best_steering_model_xy_trt.pth'))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

camera = Camera()

robot = Robot()

# TODO
speed_gain: float = 0
slow_speed_gain: float = 0
steering_gain: float = 0
steering_dgain: float = 0
steering_bias: float = 0

angle: float = 0.0
angle_last: float = 0.0


def execute(change):
    global angle, angle_last

    image = change['new']
    xy = steering_model_trt(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

    angle = np.arctan2(x, y)
    pid = angle * steering_gain + \
        (angle - angle_last) * steering_dgain
    angle_last = angle

    steering_value: float = pid+steering_bias

    print("x = {x},y = {y} ,speed = {speed},steering= {steering}".format(
        x=x, y=y, speed=speed, steering=steering_value))

    robot.left_motor.value = max(
        min(speed_gain + steering_value, 1.0), 0.0)
    robot.right_motor.value = max(
        min(speed_gain - steering_value, 1.0), 0.0)


execute({'new': camera.value})

camera.observe(execute, names='value')
