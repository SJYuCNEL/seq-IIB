from captum.attr import Saliency
from captum.attr import visualization as viz
from util import imshow
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image

def normal(data):
    _range = np.max(data)-np.min(data)
    return (data-np.min(data))/_range
#  The network we designed may output multiple values and cannot directly apply captum,
#  if an error is reported, please try to change the input dimension.
def inter(args,model,test_data,device):
    number=args.imgs_number
    ways_to_train = args.ways_to_train
    images,_,target = next(iter(test_data))
    imshow(torchvision.utils.make_grid(images[:number]))
    print('GroundTruth:',' ',f"{target[:number]}")
    images,target = images.to(device),target.to(device)
    if ways_to_train == "seq-IIB":
        outputs,_ = model(images[:number])
    elif ways_to_train == "IBIRM":
        _,outputs = model(images[:number])
    elif ways_to_train == "IIB":
        outputs = model.predict(images[:number])
    else:
        outputs = model(images[:number])
    _,predicted = torch.max(outputs,1)
    print('Predicted:',' ',f"{predicted[:number]}")

    ind = args.img_idx
    input_image = images[ind]

    input_image_0 = torch.zeros_like(input_image).to(device)
    input_image_1 = torch.zeros_like(input_image).to(device)
    # Generate red image
    input_image_0[0] = input_image[torch.argmax(torch.sum(input_image,dim=[1,2]))]
    # Generate green image
    input_image_1[1] = input_image[torch.argmax(torch.sum(input_image,dim=[1,2]))]
    
    input_0 = input_image_0.unsqueeze(0)
    input_1 = input_image_1.unsqueeze(0)
    input_0.requires_grad=True
    input_1.requires_grad=True
   
    model.eval()
    
    S= Saliency(model)
    # Generate Saliency map for red image
    grads_0 = S.attribute(input_0,target=target[ind].item())
    # Generate Saliency map for green image
    grads_1 = S.attribute(input_1,target=target[ind].item()) 
    dim_0 = np.transpose(grads_0.squeeze().cpu().detach().numpy(),(1,2,0))
    dim_1 = np.transpose(grads_1.squeeze().cpu().detach().numpy(),(1,2,0))

    print('Original Image')
    original_image = np.transpose((input_image.cpu().detach().numpy()/2)+0.5,(1,2,0))
    original_image_0 = np.transpose((input_image_0.cpu().detach().numpy()/2)+0.5,(1,2,0))
    original_image_1 = np.transpose((input_image_1.cpu().detach().numpy()/2)+0.5,(1,2,0))
    _ = viz.visualize_image_attr(None,original_image,method='original_image',title="Original Image")
    _ = viz.visualize_image_attr(None,original_image_0,method='original_image',title="Original Image")
    _ = viz.visualize_image_attr(None,original_image_1,method='original_image',title="Original Image")
    _ = viz.visualize_image_attr(None,2*dim_0,method='original_image',title="dim 0")
    _ = viz.visualize_image_attr(None,2*dim_1,method='original_image',title="dim 1")
    # Generate Saliency comparison map                            
    _ = viz.visualize_image_attr(None,2*np.abs(normal(dim_0)-normal(dim_1)),method='original_image',title="Saliency comparison map",show_colorbar=True)