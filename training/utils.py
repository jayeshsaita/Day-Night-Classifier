import torch
import matplotlib.pyplot as plt
import math


# Simple function to plot images from batch
def show_batch(x, classes, nimgs=4, mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]), denorm=True):
    if denorm: # denormalize image for plotting
      denorm_x = (x * std.view(1, 3, 1, 1)) + mean.view(1, 3, 1, 1)
    else: # Don't denormalize image, used for plotting images with single channel (Value channel for HSV image)
      denorm_x = x
    img = 0 # image counter
    nr = math.ceil(math.sqrt(nimgs)) # calculating number of rows and columns to plot
    fig, ax = plt.subplots(nrows=nr, ncols=nr, figsize=(15, 10))

    inner_break = False
    for i in range(nr):
        for j in range(nr):
            if denorm:
              ax[i][j].imshow(denorm_x[img].permute(1, 2, 0))
            else:
              ax[i][j].imshow(denorm_x[img][0])
            ax[i][j].set_title(classes[img])
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)
            img += 1
            if img == nimgs:
                inner_break = True
                break

        if inner_break:
            break