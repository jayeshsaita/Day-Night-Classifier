import torch
from torchvision import models
import cv2
import argparse


def load_model():
    # loading imagenet pretrained model from torchvision models
    mbv2 = models.mobilenet_v2(pretrained=True)
    in_features = mbv2.classifier[1].in_features
    # replacing final FC layer of pretrained model with our FC layer having output classes = 2 for day/night
    mbv2.classifier[1] = torch.nn.Linear(in_features, 2)
    # Load trained model onto CPU
    mbv2.load_state_dict(torch.load('models/mbv2_best_model.pth', map_location=torch.device('cpu')))
    # Setting model to evaluation mode
    mbv2.eval()
    return mbv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--img', required=True, help='Path to image')
    args = vars(ap.parse_args())

    # imagenet mean and std to normalize data
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    # loading model
    model = load_model()

    # reading image
    ori_img = cv2.imread(str(args['img']))
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    # resizing image to standard size
    img = cv2.resize(img, (500,500))
    # changing order of channels to (channel, height, width) format used by PyTorch
    img = torch.tensor(img).permute(2,0,1)
    # normalizing image
    img = img / 255.0
    img = (img - mean)/std
    img = img.unsqueeze(0)

    out = model(img)
    pred = torch.argmax(out)
    label = 'Day' if pred == 0 else 'Night'

    cv2.putText(ori_img, f'Prediction:{label}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    cv2.imshow('Prediction', ori_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()