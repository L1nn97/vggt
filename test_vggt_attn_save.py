import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import cv2 as cv
import numpy as np
import json
import os

@dataclass
class TestVggtAttnSaveArgs:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    attn_map_save_dir: str = "/p/project1/hai_4dgen/zcl/vggt_global_attn_maps"
    checkpoint_path: str = "/p/project1/hai_4dgen/zcl/vggt/checkpoints/model.pt"


if __name__ == "__main__":
    args = TestVggtAttnSaveArgs()

    model = VGGT(attn_map_save_dir=args.attn_map_save_dir)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(args.device)
    model.eval()
    image_names = []

    for i in range(10):
        image_names += ["./examples/kitchen/images/{:02d}.png".format(i)]

    images = load_and_preprocess_images(image_names).to(device)

    print(image_names)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
            print(type(predictions))