import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import cv2 as cv
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT()
_LOCAL_PRETRAINED_PATH = "/home/rokae/zcl/vggt/checkpoints/model.pt"
model.load_state_dict(torch.load(_LOCAL_PRETRAINED_PATH))
model.to(device)
model.eval()
# Load and preprocess example images (replace with your own image paths)
image_names = []
for i in range(10):
    image_names += ["/home/rokae/zcl/vggt/examples/kitchen/images/{:02d}.png".format(i)]

images = load_and_preprocess_images(image_names).to(device)

# row_start = int(12 * 14)
# col_start = int(18 * 14)
# row_end = row_start + 14
# col_end = col_start + 14

# image0 = images[0].cpu().numpy().transpose(1, 2, 0)
# image0 = cv.cvtColor(image0, cv.COLOR_BGR2RGB)
# image0 = cv.rectangle(image0, (col_start, row_start), (col_end, row_end), (0, 0, 255), 3)

# cv.imshow("input", image0)
# cv.waitKey(0)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        print(type(predictions))