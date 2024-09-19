import torch
import numpy as np
from PIL import Image


class BW_ConstrainImage:
    """
    A node that constrains an image to a maximum and minimum size while maintaining aspect ratio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_width": ("INT", {"default": 1024, "min": 0}),
                "max_height": ("INT", {"default": 1024, "min": 0}),
                "min_width": ("INT", {"default": 0, "min": 0}),
                "min_height": ("INT", {"default": 0, "min": 0}),
                "crop_if_required": (["yes", "no"], {"default": "no"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "constrain_image"
    CATEGORY = "不忘科技-🐱"
    OUTPUT_IS_LIST = (True,)

    def constrain_image(self, images, max_width, max_height, min_width, min_height, crop_if_required):
        crop_if_required = crop_if_required == "yes"
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")

            current_width, current_height = img.size
            aspect_ratio = current_width / current_height

            # Constrain width and height while maintaining aspect ratio
            if aspect_ratio > 1:  # Width is the constraining dimension
                constrained_width = min(current_width, max_width)
                constrained_height = int(constrained_width / aspect_ratio)
                constrained_height = max(constrained_height, min_height)
            else:  # Height is the constraining dimension
                constrained_height = min(current_height, max_height)
                constrained_width = int(constrained_height * aspect_ratio)
                constrained_width = max(constrained_width, min_width)

            # Resize image to constrained dimensions
            resized_image = img.resize((constrained_width, constrained_height), Image.LANCZOS)

            # Ensure the dimensions are even
            if constrained_width % 2 != 0:
                constrained_width -= 1
            if constrained_height % 2 != 0:
                constrained_height -= 1

            # Re-resize the image to the adjusted even dimensions
            resized_image = resized_image.resize((constrained_width, constrained_height), Image.LANCZOS)

            if crop_if_required and (constrained_width > max_width or constrained_height > max_height):
                # Calculate the center of the image
                center_width = constrained_width // 2
                center_height = constrained_height // 2

                # Calculate the crop box such that the center remains fixed
                left = max(center_width - max_width // 2, 0)
                top = max(center_height - max_height // 2, 0)
                right = min(center_width + max_width // 2, constrained_width)
                bottom = min(center_height + max_height // 2, constrained_height)

                resized_image = resized_image.crop((left, top, right, bottom))

            resized_image = np.array(resized_image).astype(np.float32) / 255.0
            resized_image = torch.from_numpy(resized_image)[None,]
            results.append(resized_image)

        return (results,)


CONSTRAIN_CLASS_MAPPINGS = {
    "BW_ConstrainImage": BW_ConstrainImage,
}

CONSTRAIN_DISPLAY_NAME_MAPPINGS = {
    "BW_ConstrainImage": "不忘科技-等比例缩放-🐱",
}
