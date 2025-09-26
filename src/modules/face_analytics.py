"""
Face detection + embedding + gallery matching pipeline.

It detects faces in images or video frames, extracts embeddings (face fingerprints), 
and matches them against a gallery of known faces to identify people.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2, numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm


@dataclass
class FaceDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    embedding: np.ndarray
    identity: Optional[str] = None
    similarity: Optional[float] = None


@dataclass
class FaceAnalyticsConfig:
    root_dir: Path = Path(__file__).resolve().parents[2]
    providers: Tuple[str, ...] = ("CPUExecutionProvider",)
    det_size: Tuple[int, int] = (640, 640)
    gallery_root: Path = root_dir / "data" / "raw" / "faces_sampled"
    gallery_cache: Path = root_dir / "data" / "processed" / "face_gallery.npz"
    similarity_threshold: float = 0.35  # cosine distance threshold
    min_score: float = 0.4
    context_id: int = 0  # GPU ID (keep 0 for CPU)


class FaceGallery:
    def __init__(self, embeddings: Dict[str, np.ndarray]) -> None:
        self.embeddings = embeddings
        self._matrix = np.stack(list(embeddings.values()), axis=0)
        self._labels = list(embeddings.keys())

    @classmethod
    def from_directory(cls, root: Path, detector: "FaceAnalytics") -> "FaceGallery":
        embeddings: Dict[str, np.ndarray] = {}
        for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            person_embeddings: List[np.ndarray] = []
            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                embedding = detector.compute_embedding(img_path)
                if embedding is not None:
                    person_embeddings.append(embedding)
            if person_embeddings:
                embeddings[person_dir.name] = np.mean(person_embeddings, axis=0)
        if not embeddings:
            raise ValueError(f"No gallery embeddings created from {root}")
        return cls(embeddings)
    
    @classmethod
    def from_cache(cls, path: Path) -> Optional["FaceGallery"]:
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        labels: List[str] = data["labels"].tolist()
        matrix: np.ndarray = data["embeddings"]
        embeddings = {label: matrix[idx] for idx, label in enumerate(labels)}
        return cls(embeddings)
    
    def to_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, embeddings=self._matrix, labels=np.array(self._labels))

    def match(self, embedding: np.ndarray, threshold: float) -> Tuple[Optional[str], Optional[float]]:
        scores = cosine_similarity(self._matrix, embedding)
        idx = int(np.argmax(scores))
        best_score = float(scores[idx])
        if best_score >= (1 - threshold):
            return self._labels[idx], best_score
        return None, None


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norm = norm(matrix, axis=1, keepdims=True) + 1e-10
    vector_norm = norm(vector) + 1e-10
    return (matrix @ vector) / (matrix_norm[:, 0] * vector_norm)


class FaceAnalytics:
    def __init__(self, config: FaceAnalyticsConfig | None = None) -> None:
        self.config = config or FaceAnalyticsConfig()
        self.app = FaceAnalysis(
            providers=list(self.config.providers),
            name="buffalo_l",
            root=Path(".cache/insightface").as_posix(),
        )
        self.app.prepare(ctx_id=self.config.context_id, det_size=self.config.det_size)
        self.gallery: Optional[FaceGallery] = None

    def load_or_build_gallery(self, root: Optional[Path] = None) -> None:
        cache_path = self.config.gallery_cache
        gallery = FaceGallery.from_cache(cache_path)
        if gallery is None:
            gallery_root = root or self.config.gallery_root
            gallery = FaceGallery.from_directory(gallery_root, self)
            gallery.to_cache(cache_path)
        self.gallery = gallery

    def compute_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        faces = self.app.get(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: float(f.det_score))
        return normalize_embedding(face.normed_embedding)

    def analyze_frame(self, frame: np.ndarray) -> List[FaceDetection]:
        faces = self.app.get(frame)
        results: List[FaceDetection] = []
        for face in faces:
            score = float(face.det_score)
            if score < self.config.min_score:
                continue
            bbox = tuple(map(float, face.bbox))
            embedding = normalize_embedding(face.normed_embedding)
            identity = None
            similarity = None
            if self.gallery is not None:
                identity, similarity = self.gallery.match(
                    embedding, threshold=self.config.similarity_threshold
                )
            results.append(
                FaceDetection(
                    bbox_xyxy=bbox,
                    score=score,
                    embedding=embedding,
                    identity=identity,
                    similarity=similarity,
                )
            )
        return results


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    embedding_norm = norm(embedding) + 1e-10
    return embedding / embedding_norm
