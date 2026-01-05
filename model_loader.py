from sentence_transformers import SentenceTransformer, models
import os
import torch

class EmbeddingModel:
    def __init__(self):
        # 1. 自动检测设备 (关键修改)
        # 如果装了 GPU 版 torch，这里会自动变成 'cuda'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using Device: {self.device.upper()}") 

        base_path = os.path.dirname(os.path.abspath(__file__))
        text_model_path = os.path.join(base_path, "models/all-MiniLM-L6-v2")
        clip_model_path = os.path.join(base_path, "models/clip-ViT-B-32")

        # 2. 加载文本模型 (传入 device 参数)
        print(f"Loading Text Model from: {text_model_path} ...")
        self.text_model = SentenceTransformer(text_model_path, device=self.device)

        # 3. 加载 CLIP 模型 (传入 device 参数)
        print(f"Loading CLIP Model from: {clip_model_path} ...")
        try:
            # 显式加载 CLIP 模块
            clip_module = models.CLIPModel(clip_model_path)
            self.clip_model = SentenceTransformer(modules=[clip_module], device=self.device)
        except Exception as e:
            print(f"标准加载失败，尝试备用方案: {e}")
            self.clip_model = SentenceTransformer(clip_model_path, device=self.device)

    def get_text_embedding(self, text):
        return self.text_model.encode(text).tolist()

    def get_image_embedding(self, image):
        return self.clip_model.encode(image).tolist()
    
    def get_text_for_image_embedding(self, text):
        return self.clip_model.encode(text).tolist()