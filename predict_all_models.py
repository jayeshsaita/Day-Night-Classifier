import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import argparse
import matplotlib.pyplot as plt

# Simple function that returns a Conv-BatchNorm-ReLU layer
def conv_bn_relu(ni, nf, stride=2, bn=True, act=True):
  layers = [nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1, bias=not bn)] # no need of bias if using batchnorm
  if bn:
    layers.append(nn.BatchNorm2d(nf))
  if act:
    layers.append(nn.ReLU(inplace=True))
  return nn.Sequential(*layers)

# Simple layer to flatten output of previous layer
class Flatten(nn.Module):
  def forward(self, x):
    return x.squeeze()

def load_simple_model():
    # Simple 5-layer FCN-CNN model that takes as input a V channel of HSV image.
    simple_model = nn.Sequential(
        conv_bn_relu(1, 8),
        conv_bn_relu(8, 16),
        conv_bn_relu(16, 32),
        conv_bn_relu(32, 8),
        conv_bn_relu(8, 2, bn=False, act=False), # no batchnorm and relu for last layer
        nn.AdaptiveAvgPool2d(1), # taking mean across spatial dimensions, these are logits
        Flatten()
    )
    # Loading trained model into CPU
    simple_model.load_state_dict(torch.load('models/simple_best_model.pth', map_location=torch.device('cpu')))
    # Setting model in eval mode
    simple_model.eval()
    return simple_model

def load_mbv2():
    # loading imagenet pretrained model from torchvision models
    mbv2 = models.mobilenet_v2(pretrained=True)
    in_features = mbv2.classifier[1].in_features
    # replacing final FC layer of pretrained model with our FC layer having output classes = 2 for day/night
    mbv2.classifier[1] = torch.nn.Linear(in_features, 2)
    # Load trained model onto CPU
    mbv2.load_state_dict(torch.load('models/mbv2_best_model.pth', map_location=torch.device('cpu')))
    # setting model to evaluation mode
    mbv2.eval()
    return mbv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--img', required=True, help='Path to image')
    args = vars(ap.parse_args())

    # transforms to normalize data for simple_model
    simple_model_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.42718], [0.22672])
    ])

    # imagenet mean and std used to normalize data for mobilenetv2
    mbv2_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    mbv2_std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    bgr_img = cv2.imread(str(args['img']))
    bgr_resized = cv2.resize(bgr_img, (500,500))

    simple_model = load_simple_model()
    hsv = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    # Prediction for Baseline model
    avg_brightness = v.mean()
    if avg_brightness > 100:
      baseline_label = 'Day'
    else:
      baseline_label = 'Night'

    # Prediction for Simple model
    v = simple_model_tfms(v)
    simple_out = simple_model(v.unsqueeze(0)).view(1,2)
    simple_probs = torch.softmax(simple_out, dim=1)
    simple_pred = torch.argmax(simple_out)
    simple_label = 'Day' if simple_pred == 1 else 'Night'

    # Prediction for MobileNetv2 model
    mbv2_model = load_mbv2()
    rgb_img = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    rgb_img = torch.tensor(rgb_img).permute(2,0,1)
    rgb_img = rgb_img / 255.0
    rgb_img = (rgb_img - mbv2_mean) / mbv2_std
    rgb_img =  rgb_img.unsqueeze(0)

    mbv2_out = mbv2_model(rgb_img).view(1,2)
    mbv2_probs = torch.softmax(mbv2_out, dim=1)
    mbv2_pred = torch.argmax(mbv2_out)
    mbv2_label = 'Day' if mbv2_pred == 0 else 'Night'

    rgb_baseline = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_simple = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_mbv2 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    cv2.rectangle(rgb_baseline, (10,10), (300, 70), (255,255,255), thickness=-1)
    cv2.rectangle(rgb_simple, (10,10), (450, 70), (255,255,255), thickness=-1)
    cv2.rectangle(rgb_mbv2, (10,10), (450, 70), (255,255,255), thickness=-1)
    cv2.putText(rgb_baseline, f'Prediction:{baseline_label}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), thickness=2)
    cv2.putText(rgb_simple, f'Prediction:{simple_label} ({(simple_probs[0][simple_pred].item() * 100):.2f}%)', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), thickness=2)
    cv2.putText(rgb_mbv2, f'Prediction:{mbv2_label} ({(mbv2_probs[0][mbv2_pred].item() * 100):.2f}%)', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), thickness=2)
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(rgb_baseline)
    ax[0].set_title('Baseline Model')
    ax[1].imshow(rgb_simple)
    ax[1].set_title('Simple Model')
    ax[2].imshow(rgb_mbv2)
    ax[2].set_title('MobileNet V2')
    plt.show()
    

if __name__ == '__main__':
    main()