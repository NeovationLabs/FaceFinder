"""
stream_worker.py

Runs against ONE video source, which can be either:
  - a live RTSP camera URL (e.g. "rtsp://user:pass@192.168.1.10:554/stream1")
  - a local video file path (e.g. "clips/person_angle1.mp4")

The source type is auto-detected: anything starting with "rtsp://" is
treated as a live stream (auto-reconnect on drop); anything else is treated
as a local file (played once, worker exits cleanly at end-of-file).

For each sampled frame:
  1. Detect all faces in the frame.
  2. Compare each detected face's embedding against the enrolled target
     embedding using cosine similarity.
  3. If similarity clears the threshold, log a match (camera id, timestamp,
     similarity score, snapshot).

This is designed to run as one process/thread per camera/video. See main.py
for how several of these are launched together and their outputs merged.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from face_engine import FaceEngine


# Cosine similarity threshold for "same person". InsightFace's buffalo_l
# embeddings: same-person pairs typically score 0.5-0.7+, different-person
# pairs typically score well under 0.3. 0.45-0.5 is a common starting point;
# TUNE THIS against your own data before relying on it (see README).
DEFAULT_MATCH_THRESHOLD = 0.45


@dataclass
class MatchEvent:
    camera_id: str
    timestamp: float
    similarity: float
    bbox: tuple
    snapshot_path: Optional[str] = None
    source_frame_index: Optional[int] = None  # useful for locating the moment in a video file


def _is_live_stream(source: str) -> bool:
    """RTSP (and similarly, rtmp/http live streams) are treated as 'live';
    anything else is assumed to be a local video file path."""
    return source.startswith(("rtsp://", "rtmp://", "http://", "https://"))


class StreamWorker:
    def __init__(
        self,
        camera_id: str,
        source: str,
        target_embedding: np.ndarray,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        sample_every_n_frames: int = 5,
        snapshot_dir: Optional[str] = None,
        loop_video: bool = False,
    ):
        """
        source: either an RTSP URL or a path to a local video file.
        loop_video: if True and source is a local file, restarts the file
                    from the beginning at end-of-file instead of exiting.
                    Has no effect on live streams (they never "end").
        """
        self.camera_id = camera_id
        self.source = source
        self.is_live = _is_live_stream(source)
        self.target_embedding = target_embedding
        self.match_threshold = match_threshold
        self.sample_every_n_frames = sample_every_n_frames
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.loop_video = loop_video
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.engine = FaceEngine.get()  # shared singleton, loaded once per process
        self.cap = None

    def connect(self):
        """
        Opens the video source. Uses FFMPEG backend explicitly since it
        handles both RTSP streams and most video file codecs more reliably
        than OpenCV's default backend.
        """
        if self.is_live:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            video_path = Path(self.source)
            if not video_path.exists():
                raise FileNotFoundError(f"[{self.camera_id}] Video file not found: {self.source}")
            self.cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            kind = "RTSP stream" if self.is_live else "video file"
            raise ConnectionError(
                f"[{self.camera_id}] Could not open {kind}: {self.source}\n"
                + (
                    "Check: correct URL/credentials, camera reachable on network, "
                    "correct RTSP port (usually 554), and that ffmpeg is installed."
                    if self.is_live
                    else "Check: file path is correct and the video codec is supported "
                         "by your installed ffmpeg."
                )
            )

    def process_frame(self, frame: np.ndarray, frame_index: int, on_match=None):
        faces = self.engine.get_faces(frame)
        for face in faces:
            similarity = self.engine.cosine_similarity(
                face.normed_embedding, self.target_embedding
            )
            if similarity >= self.match_threshold:
                snapshot_path = None
                if self.snapshot_dir:
                    ts = time.time()
                    snapshot_path = str(
                        self.snapshot_dir / f"{self.camera_id}_{int(ts)}_{frame_index}.jpg"
                    )
                    cv2.imwrite(snapshot_path, frame)

                event = MatchEvent(
                    camera_id=self.camera_id,
                    timestamp=time.time(),
                    similarity=round(similarity, 4),
                    bbox=tuple(int(v) for v in face.bbox),
                    snapshot_path=snapshot_path,
                    source_frame_index=frame_index,
                )
                if on_match:
                    on_match(event)

    def run(self, on_match=None, max_frames: Optional[int] = None):
        """
        Main loop. `on_match` is a callback invoked with each MatchEvent —
        wire this to your alerting system (dashboard push, DB write, Slack
        webhook, etc.) in main.py.

        Live streams: auto-reconnects with backoff on a dropped connection,
        since RTSP over real networks WILL occasionally drop, and runs
        forever (or until max_frames, if set).

        Local video files: reads through once and exits cleanly at
        end-of-file (unless loop_video=True, in which case it restarts from
        the beginning — useful for a looping demo without needing to
        re-launch the process).
        """
        self.connect()
        frame_index = 0
        consecutive_failures = 0

        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            ret, frame = self.cap.read()

            if not ret:
                if self.is_live:
                    consecutive_failures += 1
                    print(f"[{self.camera_id}] Frame read failed ({consecutive_failures}). "
                          f"Reconnecting...")
                    self.cap.release()
                    time.sleep(min(2 ** consecutive_failures, 30))  # backoff
                    self.connect()
                    continue
                else:
                    # End of video file reached.
                    if self.loop_video:
                        print(f"[{self.camera_id}] End of video reached, looping.")
                        self.cap.release()
                        self.connect()
                        frame_index = 0
                        continue
                    else:
                        print(f"[{self.camera_id}] End of video reached, worker finished.")
                        break

            consecutive_failures = 0
            if frame_index % self.sample_every_n_frames == 0:
                self.process_frame(frame, frame_index, on_match=on_match)
            frame_index += 1

        self.cap.release()