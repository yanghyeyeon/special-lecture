'''
 * The Tag2Text Model
 * Written by Xinyu Huang
'''
# STEP 1
import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform

# STEP 2
delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]
#######load model
model_path = "'pretrained/tag2text_swin_14m.pth'"
model = tag2text(pretrained=model_path,
                            image_size=384,
                            vit='swin_b',
                            delete_tag_index=delete_tag_index)
model.threshold = 0.68 # threshold for tagging
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# delete some tags that may disturb captioning
# 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"

# STEP 3
model = model.to(device)
image_path = "images/1641173_2291260800.jpg"
transform = get_transform(image_size=384)
image = transform(Image.open(image_path)).unsqueeze(0).to(device)

# STEP 4
res = inference(image, model)

# STEP 5
print("Model Identified Tags: ", res[0])
print("User Specified Tags: ", res[1])
print("Image Caption: ", res[2])