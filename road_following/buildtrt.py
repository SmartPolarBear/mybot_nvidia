import torchvision
import torch

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model = model.cuda().eval().half()

model.load_state_dict(torch.load('best_steering_model_xy.pth'))

device = torch.device('cuda')

from torch2trt import torch2trt

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)

torch.save(model_trt.state_dict(), 'best_steering_model_xy_trt.pth')
