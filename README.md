# Day-Night-Classifier
This repo contains code and models trained to classify day and night images.

### Requirements
- [numpy](https://pypi.org/project/numpy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [OpenCV 3](https://pypi.org/project/opencv-python/3.4.9.31/)
- [Pytorch](https://pytorch.org/get-started/locally/)
- [torchvision](https://pytorch.org/get-started/locally/)

### Dataset
The data for this project was scraped from [Pexels website](https://www.pexels.com/) using the [Download all images](https://addons.mozilla.org/en-US/firefox/addon/save-all-images-webextension/) extension for Mozilla Firefox.

The training set contains approximately 1000 images and validation set contains 200 images. An additional data cleaning phase was done manually to avoid noisy labels.

### Models
Three different approaches have been used.
- Baseline model - Basic model that uses average brightness from Value channel of HSV image as threshold to classify image. Achieves an accuracy of 88.5% on the validation set.
- Simple FCN-CNN - A Simple 5-layer Fully Convolutional Neural Network that works on Value channel of HSV image. Achieves an accuracy of 89.5% on the validation set.
- MobileNetv2 - MobileNetv2 is trained by using transfer learning from an imagenet pretrained model. Achieves an accuracy of 94.5% on the validation set.

### Files
- [baseline.ipynb](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/training/baseline.ipynb) - Training of baseline model
- [simple_hsv_model.ipynb](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/training/simple_hsv_model.ipynb) - Training of Simple 5-layer CNN model
- [mobilenetv2_transfer_learning.ipynb](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/training/mobilenetv2_transfer_learning.ipynb) - Training of MobileNetv2 model

### Inference
- [predict_simple_model.py](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/predict_simple_model.py) - Perform prediction on image using the Simple 5-layer CNN
- [predict_mbv2.py](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/predict_mbv2.py) - Perform prediction on image using the MobileNetv2 model.
- [predict_all_models.py](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/predict_all_models.py) - Performs prediction on image using all 3 models and outputs the results side by side for comparision.

#### Syntax for inference
```
python predict_file.py -i /path/to/image.jpg
```
Example:
```
python predict_all_models.py -i day_night_dataset/val/night/pexels-photo-2403202.jpeg
```

#### Sample Results
These sample results are generated using the [predict_all_models.py](https://github.com/jayeshsaita/Day-Night-Classifier/blob/master/predict_all_models.py) file.

![result1.png](result1.png)
![result2.png](result2.png)
![result5.png](result5.png)
![result4.png](result4.png)
![result6.png](result6.png)
![result3.png](result3.png)
