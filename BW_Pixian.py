import io
import numpy as np
from PIL import Image
import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BW_PixianBackgroundRemover:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 使用 IMAGE 作为输入类型
            },
        }

    RETURN_TYPES = ("IMAGE",)  # 定义返回类型为 IMAGE
    FUNCTION = "remove_background"  # 指定节点的执行函数
    CATEGORY = "不忘科技-🐱"  # 分类为图像处理

    API_KEY = 'px5qsvedw3xvqld'  # 你的API密钥
    API_SECRET = 'bho9j59bs0go9finoldnhfmbg2cdniv03psqauehlqouvceta2vl'  # 你的API密钥

    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def remove_background(self, image):
        # 如果 image 是一个 PyTorch Tensor
        if isinstance(image, torch.Tensor):
            # 确保 Tensor 在 CPU 上
            image = image.cpu()

            # 如果 Tensor 是四维的，选择第一个图像（批处理中的第一张照片）
            if image.dim() == 4:
                image = image[0]

            # 转换 Tensor 的数据类型到 uint8，范围从 [0, 1] 转换到 [0, 255]
            image = image * 255
            if image.dtype != torch.uint8:
                image = image.byte()

            # 将 Tensor 转换为 NumPy 数组
            image_np = image.numpy()
        elif isinstance(image, np.ndarray):
            # 如果 image 已经是一个 NumPy 数组，直接使用
            image_np = image
        else:
            raise TypeError("Unsupported image type")

        # 确保图像是三维数组
        if image_np.ndim != 3:
            raise ValueError("Image array must be 3-dimensional")

        # 将 NumPy 数组转换为 PIL Image 对象
        image = Image.fromarray(image_np)

        # 将 PIL 图像转换为字节流
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)  # 重置指针到字节流的开始位置

        # 创建一个 Session 对象
        session = requests.Session()

        # 设置重试策略
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        output_image = None  # 初始化 output_image

        try:
            # 使用 session 对象发送 POST 请求
            response = session.post(
                'https://api.pixian.ai/api/v2/remove-background',
                files={'image': ('input.png', image_bytes, 'image/png')},
                auth=(self.API_KEY, self.API_SECRET),
                timeout=10  # 设置10秒超时
            )
        except requests.exceptions.RequestException as e:
            print(f"请求失败，无法重试: {e}")
            return [None]

        # 检查响应状态
        if response.status_code == 200:
            output_image_bytes = io.BytesIO(response.content)
            # 使用 'RGBA' 模式打开图像，以保留透明度信息
            output_image = Image.open(output_image_bytes).convert('RGBA')

            tensor_image = self.pil2tensor(output_image)

            print("背景移除成功")
            return [tensor_image]  # 返回 Tensor 列表
        else:
            print(f"Error removing background: {response.status_code}, {response.text}")
            return [None]  # 如果发生错误，返回一个包含 None 的列表


# 定义节点类映射和显示名称映射
PIXIAN_CLASS_MAPPINGS = {
    "BW_PixianBackgroundRemover": BW_PixianBackgroundRemover
}

PIXIAN_DISPLAY_NAME_MAPPINGS = {
    "BW_PixianBackgroundRemover": "不忘科技-抠图-PiXian调用-🐱"
}
