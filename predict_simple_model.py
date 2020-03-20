import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import argparse


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

def load_model():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--img', required=True, help='Path to image')
    args = vars(ap.parse_args())

    # transforms to normalize data
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.42718], [0.22672])
    ])

    # loading model
    model = load_model()

    # reading image
    img = cv2.imread(str(args['img']))
    # resizing image to standard size
    resized = cv2.resize(img, (500,500))
    # converting image to HSV colorspace
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    # splitting channels of HSV image
    _, _, v = cv2.split(hsv)
    # normalizing Value channel of HSV image
    v = tfms(v)

    out = model(v.unsqueeze(0))
    pred = torch.argmax(out)
    label = 'Day' if pred == 1 else 'Night'

    cv2.putText(img, f'Prediction:{label}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    cv2.imshow('Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()