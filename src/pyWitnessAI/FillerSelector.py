import os
import torch
import clip
import pickle
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from facenet_pytorch import MTCNN

class CLIPFillerSelector:
    def __init__(self, image_dir: str, cache_dir: str = './cache', device: str = None):
        """
        :param image_dir: 存放候选照片的目录
        :param cache_dir: 特征缓存目录，可序列化大规模图库
        :param device:    'cuda' 或 'cpu'，默认自动检测
        """
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 CLIP 模型和预处理
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # 初始化人脸检测与对齐
        self.face_detector = MTCNN(image_size=224, margin=20, keep_all=False, device=self.device)

        # 构建原始图像路径列表
        self.original_paths = [
            os.path.join(image_dir, fn)
            for fn in os.listdir(image_dir)
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        # 缓存文件路径
        feat_file = os.path.join(self.cache_dir, 'image_features.pt')
        paths_file = os.path.join(self.cache_dir, 'image_paths.pkl')

        # 尝试从缓存加载
        if os.path.exists(feat_file) and os.path.exists(paths_file):
            self.image_features = torch.load(feat_file, map_location=self.device)
            with open(paths_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            # 检查缓存一致性
            if len(self.image_paths) != self.image_features.shape[0]:
                print("[Warning] 缓存不一致，重建特征缓存...")
                self._build_cache(feat_file, paths_file)
        else:
            # 初次或缓存缺失，构建缓存
            self._build_cache(feat_file, paths_file)

    def _build_cache(self, feat_file, paths_file):
        """重建图像特征缓存并序列化到磁盘。"""
        self._cache_image_features()
        torch.save(self.image_features, feat_file)
        with open(paths_file, 'wb') as f:
            pickle.dump(self.image_paths, f)

    def _cache_image_features(self):
        """检测、对齐、预处理并提取所有图像的 CLIP 特征。"""
        valid_paths = []
        all_feats = []
        batch_imgs = []
        batch_paths = []
        batch_size = 32

        for path in tqdm(self.original_paths, desc="Detecting & aligning faces"):
            try:
                img = Image.open(path).convert("RGB")
                face_tensor = self.face_detector(img)
                if face_tensor is None:
                    continue
                face_pil = Image.fromarray(
                    (face_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                )
                batch_imgs.append(self.preprocess(face_pil))
                batch_paths.append(path)
            except Exception:
                continue

            if len(batch_imgs) >= batch_size:
                feats = self._encode_batch(batch_imgs)
                all_feats.append(feats)
                valid_paths.extend(batch_paths)
                batch_imgs.clear()
                batch_paths.clear()

        # 处理剩余批次
        if batch_imgs:
            feats = self._encode_batch(batch_imgs)
            all_feats.append(feats)
            valid_paths.extend(batch_paths)

        # 更新特征和路径
        self.image_features = torch.cat(all_feats, dim=0)
        self.image_paths = valid_paths

    def _encode_batch(self, img_tensors):
        batch = torch.stack(img_tensors, dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = F.normalize(feats, dim=-1)
        return feats

    def query(self, verbal_desc: str, top_k: int = 5):
        """
        根据口头描述检索最相似的图像。
        :param verbal_desc: 目击者描述
        :param top_k:       返回 Top-K 照片数量
        :return:            List[(image_path, score)]
        """
        text_tokens = clip.tokenize([verbal_desc], truncate=True).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(text_tokens)
            text_feat = F.normalize(text_feat, dim=-1)

        scores = (self.image_features @ text_feat.T).squeeze(1)
        vals, idxs = scores.topk(top_k, largest=True)
        return [(self.image_paths[i], float(vals[j].item())) for j, i in enumerate(idxs)]

# —— 使用示例 —— #
if __name__ == "__main__":
    selector = CLIPFillerSelector(
        image_dir="D:/PhD/Studies/VerbalDescription/FillerPoolColloff", cache_dir="./cache"
    )
    matches = selector.query("Mid-age male with short hair", top_k=1)
    for img_path, score in matches:
        print(f"{img_path}  —— Similarity Score：{score:.4f}")

