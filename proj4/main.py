# STEP 0 : prepare
# 1. install git
# 2. install package  : pip install git+https://github.com/xinyu1205/recognize-anything.git
# 3. model download : https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth

# STEP 1 : import modules
import numpy as np
import random
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

# STEP 2: create inference object
model_path = "ram_plus_swin_large_14m.pth"
model = ram_plus(pretrained=model_path,
                            image_size=384,
                            vit='swin_l')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# STEP 3: Load data
image_path = 'images/demo/demo1.jpg'
transform = get_transform(image_size=384)
image = transform(Image.open(image_path)).unsqueeze(0).to(device)

# STEP 4: inference
res = inference(image, model)

# STEP 5: post processing
print("Image Tags: ", res[0])