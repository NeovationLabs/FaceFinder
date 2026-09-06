"""
face_engine.py

Thin wrapper around InsightFace's buffalo_l model, which does BOTH face
detection and embedding extraction in one pass. This is the same class of
model used in commercial face-search systems.

- Detection: finds face bounding boxes in a frame.
- Embedding: turns each detected face into a 512-dim vector. Two faces of
  the same person produce vectors that are close together (cosine similarity
  near 1.0); different people produce vectors that are far apart.

We load ONE shared instance of this engine and reuse it across the reference
photo and every camera stream, so "is this the same person" is just a vector
comparison — no separate model per camera.
"""

import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    _instance = None  # simple singleton so the model loads only once per process

    def __init__(self, det_size=(640, 640), providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = FaceEngine()
        return cls._instance

    def get_faces(self, frame: np.ndarray):
        """
        Returns a list of insightface Face objects for every face detected
        in the frame. Each has `.bbox` (x1,y1,x2,y2) and `.normed_embedding`
        (unit-normalized 512-dim vector).
        """
        return self.app.get(frame)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both are already L2-normalized by insightface