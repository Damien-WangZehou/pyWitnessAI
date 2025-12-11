import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import io
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import pandas as pd
from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from .Images import Images

class ImagesAI:
    def __init__(
        self,
        column_images: Images,
        row_images: Images,
        distance_metric: str = "euclidean",
        backend: str = "opencv",
        enforce_detection: bool = False,
        model: str = "VGG-Face",
        align: bool = False,
        normalization: str = "base",
        show_progress: bool = False,
        device: str = None,
    ):
        self.column_images = column_images
        self.row_images = row_images
        self.distance_metric = distance_metric
        self.backend = backend
        self.enforce_detection = enforce_detection
        self.model = model
        self.align = align
        self.normalization = normalization
        self.show_progress = show_progress

        # results
        self.similarity_matrix: pd.DataFrame | None = None
        self.method_used: str | None = None

        # Torch device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # FaceNet stack
        self.mtcnn = MTCNN(device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    # ---------- utils ----------
    def _progress(self, iterable, desc: str):
        # Lazy import to keep dependency light if user never wants bars
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, disable=not self.show_progress)

    # ---------- embeddings ----------
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        DeepFace.represent wrapper. Returns 1D numpy embedding.
        """
        image_array = np.array(image)  # HWC RGB

        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            reps = DeepFace.represent(
                image_array,
                model_name=self.model,
                enforce_detection=self.enforce_detection,
                detector_backend=self.backend,
                align=self.align,
                normalization=self.normalization
            )

        if not reps:
            raise ValueError("DeepFace.represent returned no embeddings.")
        return np.asarray(reps[0]['embedding'], dtype=np.float32)

    def get_embedding_facenet(self, img: Image.Image) -> torch.Tensor | None:
        """
        MTCNN detect+align -> InceptionResnetV1 embedding.
        Returns 1xD torch tensor on CPU for consistent downstream math.
        """
        with torch.no_grad():
            face = self.mtcnn(img)
            if face is None:
                return None
            emb = self.resnet(face.unsqueeze(0).to(self.device)).cpu()
        return emb.squeeze(0)  # shape [D]

    # ---------- distances ----------
    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    def calculate_similarity_euclidean(self, embedding1, embedding2) -> float:
        a = self._to_numpy(embedding1).astype(np.float32)
        b = self._to_numpy(embedding2).astype(np.float32)
        return float(np.linalg.norm(a - b))

    def calculate_similarity_euclidean_l2(self, embedding1, embedding2) -> float:
        a = self._to_numpy(embedding1).astype(np.float32)
        b = self._to_numpy(embedding2).astype(np.float32)
        a /= (np.linalg.norm(a) + 1e-12)
        b /= (np.linalg.norm(b) + 1e-12)
        return float(np.linalg.norm(a - b))

    def calculate_similarity_cosine(self, embedding1, embedding2) -> float:
        a = self._to_numpy(embedding1).astype(np.float32)
        b = self._to_numpy(embedding2).astype(np.float32)
        a /= (np.linalg.norm(a) + 1e-12)
        b /= (np.linalg.norm(b) + 1e-12)
        return float(1.0 - np.dot(a, b))

    def _distance(self, e1, e2) -> float:
        if self.distance_metric == "euclidean":
            return self.calculate_similarity_euclidean(e1, e2)
        if self.distance_metric == "euclidean_l2":
            return self.calculate_similarity_euclidean_l2(e1, e2)
        if self.distance_metric == "cosine":
            return self.calculate_similarity_cosine(e1, e2)
        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    # ---------- pipelines ----------
    def process(self):
        """
        Embedding-based similarities using DeepFace.represent.
        """
        column_embeddings = {}
        row_embeddings = {}

        for k, img in self._progress(list(self.column_images.images.items()), "Embeddings (columns)"):
            try:
                column_embeddings[k] = self.get_embedding(img)
            # Don't fail the whole process if one image fails, so that VideoLineupPipeline can continue
            except Exception:
                column_embeddings[k] = None

        for k, img in self._progress(list(self.row_images.images.items()), "Embeddings (rows)"):
            try:
                row_embeddings[k] = self.get_embedding(img)
            except Exception:
                row_embeddings[k] = None

        similarity_data = []
        col_keys = list(column_embeddings.keys())
        row_keys = list(row_embeddings.keys())

        for r_key in self._progress(row_keys, "Distances"):
            r_emb = row_embeddings[r_key]
            row_scores = []
            for c_key in col_keys:
                c_emb = column_embeddings[c_key]
                if (r_emb is None) or (c_emb is None):
                    row_scores.append(np.nan)
                else:
                    row_scores.append(self._distance(r_emb, c_emb))
            similarity_data.append(row_scores)

        self.similarity_matrix = pd.DataFrame(
            similarity_data,
            index=row_keys,
            columns=col_keys
        ).astype(float)
        self.method_used = "process"

    def process_verify(self):
        """
        Similarities via DeepFace.verify (distance output).
        """
        rows = list(self.row_images.images.items())
        cols = list(self.column_images.images.items())

        similarity_data = []
        for r_key, r_img in self._progress(rows, "Verify (rows)"):
            r_img_np = np.array(r_img)
            row_scores = []
            for c_key, c_img in cols:
                try:
                    c_img_np = np.array(c_img)
                    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                        result = DeepFace.verify(
                            r_img_np,
                            c_img_np,
                            model_name=self.model,
                            detector_backend=self.backend,
                            enforce_detection=self.enforce_detection,
                            distance_metric=self.distance_metric,
                            align=self.align,
                            normalization=self.normalization
                        )
                    row_scores.append(float(result['distance']))
                except Exception:
                    row_scores.append(np.nan)
            similarity_data.append(row_scores)

        self.similarity_matrix = pd.DataFrame(
            similarity_data,
            index=[k for k, _ in rows],
            columns=[k for k, _ in cols]
        ).astype(float)
        self.method_used = "process_verify"

    def process_georgia_pipeline(self):
        """
        MTCNN + InceptionResnetV1 (Facenet) embeddings (as described in Kleider-Offutt et al. 2024).
        """
        column_embeddings = {}
        row_embeddings = {}

        for k, img in self._progress(list(self.column_images.images.items()), "FaceNet (columns)"):
            column_embeddings[k] = self.get_embedding_facenet(img)

        for k, img in self._progress(list(self.row_images.images.items()), "FaceNet (rows)"):
            row_embeddings[k] = self.get_embedding_facenet(img)

        similarity_data = []
        for r_key, r_emb in self._progress(list(row_embeddings.items()), "Distances"):
            row_scores = []
            for c_key, c_emb in column_embeddings.items():
                if r_emb is None or c_emb is None:
                    row_scores.append(np.nan)
                else:
                    row_scores.append(self.calculate_similarity_euclidean(r_emb, c_emb))
            similarity_data.append(row_scores)

        self.similarity_matrix = pd.DataFrame(
            similarity_data,
            index=row_embeddings.keys(),
            columns=column_embeddings.keys()
        ).astype(float)
        self.method_used = "process_georgia_pipeline"

    # ---------- io ----------
    def dataframe(self) -> pd.DataFrame:
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix. Run one of the process*() methods first.")
        return self.similarity_matrix.round(4)

    def save(self, directory='results', label='similarity_matrix'):
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix to save. Run process*() first.")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{label}.csv")
        self.similarity_matrix.to_csv(path, index=True)
        print(f"Similarity matrix saved to {path}.")