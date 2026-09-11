import os
import pickle
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class CLIPFillerSelector:
    """Legacy largest-face CLIP selector.

    New folder-search workflows should use :class:`pyWitnessAI.FaceSearch`.
    This class keeps its historical largest-face cropping and tuple-list return
    format so existing experiments do not silently change behaviour.
    """

    def __init__(self, image_dir: str, cache_dir: str = './cache', device: str = None):
        """
        :param image_dir: The directory of filler pool
        :param cache_dir: The directory to store the features
        :param device:    'cuda' / 'cpu'，sentence-transformers auto-set.
        """
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Use sentence-transformers to load the model
        self.model = SentenceTransformer('clip-ViT-B-32', device=device)

        # Use haar to detect faces
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Build the path of original images
        self.original_paths = [
            os.path.join(image_dir, fn)
            for fn in os.listdir(image_dir)
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        feat_file = os.path.join(self.cache_dir, 'image_features.npy')
        paths_file = os.path.join(self.cache_dir, 'image_paths.pkl')

        if os.path.exists(feat_file) and os.path.exists(paths_file):
            self.image_features = np.load(feat_file)
            with open(paths_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            if len(self.image_paths) != self.image_features.shape[0]:
                print("[Warning] Cache files are not consistent, rebuilding...")
                self._build_cache(feat_file, paths_file)
        else:
            self._build_cache(feat_file, paths_file)

    def _build_cache(self, feat_file: str, paths_file: str):
        """Build the cache of image features and paths."""
        self._cache_image_features()
        np.save(feat_file, self.image_features)
        with open(paths_file, 'wb') as f:
            pickle.dump(self.image_paths, f)
        print(f"[Info] Cache files built, {len(self.image_paths)} images cached.")

    def _detect_and_crop_face(self, img_pil: Image.Image):
        """Use OpenCV to detect and crop the face."""
        img_np = np.array(img_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None

        # Get the face with the largest area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = int(min(w, h) * 0.15)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_np.shape[1], x + w + margin)
        y2 = min(img_np.shape[0], y + h + margin)
        return img_pil.crop((x1, y1, x2, y2)).resize((224, 224))

    def _cache_image_features(self):
        """Detect and crop faces, then cache the features."""
        valid_paths = []
        valid_images = []

        for path in tqdm(self.original_paths, desc="Detecting & cropping faces"):
            try:
                img = Image.open(path).convert('RGB')
                face = self._detect_and_crop_face(img)
                if face is None:
                    continue
                valid_images.append(face)
                valid_paths.append(path)
            except Exception as e:
                print(f"[Skip] {path}: {e}")
                continue

        if not valid_images:
            print("[Warning] No faces detected, feature cache is empty.")
            self.image_features = np.zeros((0, 512), dtype=np.float32)
            self.image_paths = []
            return

        # Batch encode
        batch_size = 32
        all_feats = []
        for i in tqdm(range(0, len(valid_images), batch_size), desc="Encoding features"):
            batch = valid_images[i:i + batch_size]
            feats = self.model.encode(
                batch, batch_size=batch_size,
                convert_to_numpy=True, show_progress_bar=False
            )
            # L2 normalize
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            all_feats.append(feats)

        self.image_features = np.vstack(all_feats).astype(np.float32)
        self.image_paths = valid_paths

    def query(self, verbal_desc: str, top_k: int = 5):
        """
        Search for the most similar images based on the verbal description.
        :param verbal_desc: Verbal description of eyeWitness
        :param top_k:       Return top-k similar images
        :return:            List[(image_path, score)]
        """
        if self.image_features.shape[0] == 0:
            print("[Warning] Feature cache is empty, no images to search.")
            return []

        text_feat = self.model.encode([verbal_desc], convert_to_numpy=True)
        text_feat = text_feat / np.linalg.norm(text_feat, axis=1, keepdims=True)

        scores = (self.image_features @ text_feat.T).squeeze(1)
        top_k = min(top_k, len(self.image_paths))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.image_paths[i], float(scores[i])) for i in top_indices]


# Usage example
if __name__ == "__main__":
    selector = CLIPFillerSelector(
        image_dir="D:/PhD/Studies/VerbalDescription/FillerPoolColloff",
        cache_dir="./cache"
    )
    matches = selector.query("Mid-age male with short hair", top_k=1)
    for img_path, score in matches:
        print(f"{img_path}  —— Similarity Score：{score:.4f}")
