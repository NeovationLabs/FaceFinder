"""
enroll_reference.py

Step 1 of the pipeline: takes a photo of the person you want to find and
turns it into a face embedding ("faceprint") that every camera stream will
compare against.

Usage:
    python src/enroll_reference.py --photo reference_photos/person.jpg
"""

import argparse
import pickle
from pathlib import Path

import cv2

from face_engine import FaceEngine


def enroll(photo_path: str, output_path: str):
    image = cv2.imread(photo_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {photo_path}")

    engine = FaceEngine.get()
    faces = engine.get_faces(image)

    if len(faces) == 0:
        raise ValueError(
            "No face detected in the reference photo. Use a clear, "
            "front-facing, well-lit photo of just the target person."
        )
    if len(faces) > 1:
        print(f"Warning: {len(faces)} faces detected in reference photo. "
              f"Using the largest face (assumed to be the main subject).")
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    target_face = faces[0]
    embedding = target_face.normed_embedding

    with open(output_path, "wb") as f:
        pickle.dump({"embedding": embedding, "source_photo": photo_path}, f)

    print(f"Enrolled reference face from {photo_path}")
    print(f"Saved faceprint to {output_path}")
    return embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True, help="Path to the reference photo")
    parser.add_argument(
        "--out", default=str(Path(__file__).parent.parent / "reference_photos" / "target.faceprint"),
        help="Where to save the encoded faceprint",
    )
    args = parser.parse_args()
    enroll(args.photo, args.out)