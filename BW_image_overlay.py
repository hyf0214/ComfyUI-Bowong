import torch
import numpy as np  # 导入 NumPy 库，并使用别名 np
from PIL import Image, ImageOps
import comfy.utils
import random

# 定义最大分辨率常量
MAX_RESOLUTION = 48000


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def random_offset(max_offset):
    # ""“随机生成偏移量，但不超过画布边界”""
    return random.randint(0, max_offset)


class BW_ImageOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "coverage_method": (["基础覆盖", "随机覆盖"],),
                "overlay_resize": (["None", "Fit", "Resize by rescale_factor", "Resize to width & heigth"],),
                "resize_method": (["nearest-exact", "bilinear", "area"],),
                "rescale_factor": ("FLOAT", {"default": 1, "min": 0.01, "max": 16.0, "step": 0.1}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "x_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "y_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "rotation": ("INT", {"default": 0, "min": -180, "max": 180, "step": 5}),
                "opacity": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 5}),
            },
            "optional": {"optional_mask": ("MASK",), }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay_image"
    CATEGORY = "不忘科技-🐱"

    def apply_overlay_image(self, base_image, overlay_image, coverage_method, overlay_resize, resize_method,
                            rescale_factor,
                            width, height, x_offset, y_offset, rotation, opacity, optional_mask=None):
        # Pack tuples and assign variables
        size = width, height
        location = x_offset, y_offset
        mask = optional_mask

        if coverage_method == "基础覆盖":
            # Extract overlay_image size and store in Tuple "overlay_image_size" (WxH)
            overlay_image_size = overlay_image.size()
            overlay_image_size = (overlay_image_size[2], overlay_image_size[1])
            if overlay_resize == "Fit":
                h_ratio = base_image.size()[1] / overlay_image_size[1]
                w_ratio = base_image.size()[2] / overlay_image_size[0]
                ratio = min(h_ratio, w_ratio)
                overlay_image_size = tuple(round(dimension * ratio) for dimension in overlay_image_size)
            elif overlay_resize == "Resize by rescale_factor":
                overlay_image_size = tuple(int(dimension * rescale_factor) for dimension in overlay_image_size)
            elif overlay_resize == "Resize to width & heigth":
                overlay_image_size = (size[0], size[1])

            samples = overlay_image.movedim(-1, 1)
            overlay_image = comfy.utils.common_upscale(samples, overlay_image_size[0], overlay_image_size[1],
                                                       resize_method, False)
            overlay_image = overlay_image.movedim(1, -1)

            overlay_image = tensor2pil(overlay_image)

            # Add Alpha channel to overlay
            overlay_image = overlay_image.convert('RGBA')
            overlay_image.putalpha(Image.new("L", overlay_image.size, 255))

            # If mask connected, check if the overlay_image image has an alpha channel
            if mask is not None:
                # Convert mask to pil and resize
                mask = tensor2pil(mask)
                mask = mask.resize(overlay_image.size)
                # Apply mask as overlay's alpha
                overlay_image.putalpha(ImageOps.invert(mask))

            # Rotate the overlay image
            overlay_image = overlay_image.rotate(rotation, expand=True)

            # Apply opacity on overlay image
            r, g, b, a = overlay_image.split()
            a = a.point(lambda x: max(0, int(x * (1 - opacity / 100))))
            overlay_image.putalpha(a)

            # Split the base_image tensor along the first dimension to get a list of tensors
            base_image_list = torch.unbind(base_image, dim=0)

            processed_base_image_list = []
            for tensor in base_image_list:
                # Convert tensor to PIL Image
                image = tensor2pil(tensor)

                # Paste the overlay image onto the base image
                if mask is None:
                    image.paste(overlay_image, location)
                else:
                    image.paste(overlay_image, location, overlay_image)

                # Convert PIL Image back to tensor
                processed_tensor = pil2tensor(image)

                # Append to list
                processed_base_image_list.append(processed_tensor)

            # Combine the processed images back into a single tensor
            base_image = torch.stack([tensor.squeeze() for tensor in processed_base_image_list])

            # Return the edited base image
            return (base_image,)

        elif coverage_method == "随机覆盖":
            base_image1 = tensor2pil(base_image)
            foreground_image = tensor2pil(overlay_image).convert('RGBA')
            # 读取背景图像尺寸
            bg_width, bg_height = base_image1.size

            # 设置画布尺寸与背景图尺寸相同
            canvas_width, canvas_height = bg_width, bg_height

            # 放大背景图10%，并创建新图像
            new_bg_width = int(bg_width * 1.1)
            new_bg_height = int(bg_height * 1.1)
            enlarged_background_image = base_image1.resize((new_bg_width, new_bg_height), Image.LANCZOS)

            # 计算可移动的最大距离
            max_offset_x = new_bg_width - bg_width
            max_offset_y = new_bg_height - bg_height

            # 随机选择偏移量
            offset_x = random_offset(max_offset_x)
            offset_y = random_offset(max_offset_y)

            # 确保背景图在放大后不会移出画布范围
            offset_x = max(0, min(offset_x, bg_width))
            offset_y = max(0, min(offset_y, bg_height))

            # 移动背景图像
            moved_background_image = enlarged_background_image.crop(
                (offset_x, offset_y, offset_x + bg_width, offset_y + bg_height)
            )

            # 创建最终的合成图像
            overlay_image = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            overlay_image.paste(moved_background_image, (0, 0))
            # 确保前景图像没有透明通道，或者在粘贴前转换为 RGBA
            overlay_image.paste(foreground_image, (0, 0), foreground_image)

            # 保存最终的图像前转换为 'RGB' 模式
            if overlay_image.mode == 'RGBA':
                overlay_image = overlay_image.convert('RGB')

            overlay_image.putalpha(Image.new("L", overlay_image.size, 255))

            if optional_mask is not None:
                # 将蒙版转换为 pil 并调整大小
                mask = tensor2pil(optional_mask)
                mask = mask.resize(overlay_image.size)
                # 应用遮罩作为叠加层的 Alpha
                overlay_image.putalpha(ImageOps.invert(mask))

            # 旋转叠加图像
            overlay_image = overlay_image.rotate(rotation, expand=True)

            # Apply opacity on overlay image
            r, g, b, a = overlay_image.split()
            a = a.point(lambda x: max(0, int(x * (1 - opacity / 100))))
            overlay_image.putalpha(a)
            overlay_image1 = pil2tensor(moved_background_image)
            # Split the base_image tensor along the first dimension to get a list of tensors
            base_image_list = torch.unbind(overlay_image1, dim=0)

            processed_base_image_list = []
            for tensor in base_image_list:
                # Convert tensor to PIL Image
                image = tensor2pil(tensor)

                # Paste the overlay image onto the base image
                if optional_mask is None:
                    image.paste(overlay_image, (x_offset, y_offset))
                else:
                    image.paste(overlay_image, (x_offset, y_offset), overlay_image)

                # Convert PIL Image back to tensor
                processed_tensor = pil2tensor(image)

                # Append to list
                processed_base_image_list.append(processed_tensor)

            # Combine the processed images back into a single tensor
            base_image = torch.stack([tensor.squeeze() for tensor in processed_base_image_list])

            # Return the edited base image
            return (base_image,)


OVERLAY_CLASS_MAPPINGS = {
    "BW_ImageOverlay": BW_ImageOverlay
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
OVERLAY_DISPLAY_NAME_MAPPINGS = {
    "BW_ImageOverlay": "不忘科技-随机位置覆盖图像-🐱"
}
