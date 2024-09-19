import io
import urllib.request

import numpy as np
import torch
from PIL import Image


class URLImageLoader:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("STRING", {"default": "", "description": "一行url对应一张图", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "不忘科技-🐱"
    OUTPUT_IS_LIST = (True,)

    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def load(self, urls):
        url_list = []
        with io.StringIO(urls) as buffer:
            for line in buffer.readlines():
                url_list.append(line)
        tensors = []
        for url in url_list:
            filename, headers = urllib.request.urlretrieve(url)
            image = Image.open(filename)
            urllib.request.urlcleanup()
            tensors.append(self.pil2tensor(image))
        return [tensors]


Image_CLASS_MAPPINGS = {
    "BW_URLImageLoader": URLImageLoader
}

Image_DISPLAY_NAME_MAPPINGS = {
    "BW_URLImageLoader": "不忘科技-URL加载图片-🐱"
}
