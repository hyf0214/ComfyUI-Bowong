import random


class RandomSeed:
    def __init__(self):
        pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "gen"
    CATEGORY = "不忘科技-🐱"

    def gen(self):
        return [random.randint(1, 4294967294)]


RandomSeed_CLASS_MAPPINGS = {
    "BW_RandomSeed": RandomSeed
}

RandomSeed_DISPLAY_NAME_MAPPINGS = {
    "BW_RandomSeed": "不忘科技-随机种子-🐱"
}
