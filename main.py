"""
main.py

Launches one StreamWorker per camera/video (as a separate process, so a
crash or stall on one source doesn't take down the others) and collects
match events from all of them into a single alert log.

Each entry in cameras.yaml can point to EITHER a live RTSP stream OR a local
video file — StreamWorker auto-detects which based on the "source" value.
You can freely mix both in the same config (e.g. 3 live cameras + 1 test
video file).

Usage:
    python src/main.py --config cameras.yaml --faceprint reference_photos/target.faceprint

cameras.yaml example (live RTSP):
    cameras:
      - id: "entrance"
        source: "rtsp://user:pass@192.168.1.10:554/stream1"
      - id: "checkout_1"
        source: "rtsp://user:pass@192.168.1.11:554/stream1"

cameras.yaml example (local video files, e.g. for a demo):
    cameras:
      - id: "angle_1"
        source: "clips/person_front.mp4"
      - id: "angle_2"
        source: "clips/person_left.mp4"
      - id: "angle_3"
        source: "clips/person_right.mp4"
      - id: "angle_4"
        source: "clips/person_back.mp4"
      loop_video: true   # optional, top-level: restart each file at its end
"""

import argparse
import json
import pickle
import time
from multiprocessing import Process, Queue
from pathlib import Path

import yaml

from stream_worker import StreamWorker


def worker_process(camera_id: str, source: str, target_embedding, threshold: float,
                    snapshot_dir: str, event_queue: Queue, loop_video: bool = False):
    """Entry point run inside each camera/video's own process."""

    def on_match(event):
        event_queue.put({
            "camera_id": event.camera_id,
            "timestamp": event.timestamp,
            "similarity": event.similarity,
            "bbox": event.bbox,
            "snapshot_path": event.snapshot_path,
            "source_frame_index": event.source_frame_index,
        })

    worker = StreamWorker(
        camera_id=camera_id,
        source=source,
        target_embedding=target_embedding,
        match_threshold=threshold,
        snapshot_dir=snapshot_dir,
        loop_video=loop_video,
    )
    print(f"[{camera_id}] Starting worker for source: {source}")
    worker.run(on_match=on_match)


def load_config(config_path: str):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_target_embedding(faceprint_path: str):
    with open(faceprint_path, "rb") as f:
        data = pickle.load(f)
    return data["embedding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to cameras.yaml")
    parser.add_argument("--faceprint", required=True, help="Path to enrolled .faceprint file")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--snapshot-dir", default=str(Path(__file__).parent.parent / "logs" / "snapshots"))
    parser.add_argument("--alert-log", default=str(Path(__file__).parent.parent / "logs" / "alerts.jsonl"))
    args = parser.parse_args()

    config = load_config(args.config)
    target_embedding = load_target_embedding(args.faceprint)

    # Optional top-level flag in cameras.yaml: loop_video: true
    # Applies to any entry whose source is a local video file (ignored for
    # live RTSP entries, which never "end").
    default_loop_video = config.get("loop_video", False)

    event_queue = Queue()
    processes = []

    for cam in config["cameras"]:
        # Accept "source" (current) or "rtsp_url" (older configs) for the
        # video source field, so existing cameras.yaml files still work.
        source = cam.get("source") or cam.get("rtsp_url")
        if not source:
            raise ValueError(
                f"Camera entry {cam.get('id', '<unnamed>')} is missing a "
                f"'source' field (RTSP URL or video file path)."
            )
        loop_video = cam.get("loop_video", default_loop_video)

        p = Process(
            target=worker_process,
            args=(cam["id"], source, target_embedding, args.threshold,
                  args.snapshot_dir, event_queue, loop_video),
            daemon=True,
        )
        p.start()
        processes.append(p)
        print(f"Launched worker for '{cam['id']}' -> {source} (PID {p.pid})")

    print(f"\nAll {len(processes)} worker(s) running. Watching for matches...\n")
    print(f"Alerts will be appended to: {args.alert_log}\n")

    try:
        with open(args.alert_log, "a") as log_file:
            while True:
                try:
                    # Short timeout so we periodically get a chance to check
                    # whether all workers have finished (relevant for local
                    # video files, which end; live RTSP workers never do).
                    event = event_queue.get(timeout=1.0)
                except Exception:
                    event = None

                if event is not None:
                    print(f"[MATCH] '{event['camera_id']}' at "
                          f"{time.strftime('%H:%M:%S', time.localtime(event['timestamp']))} "
                          f"(similarity={event['similarity']}, frame={event['source_frame_index']})")
                    log_file.write(json.dumps(event) + "\n")
                    log_file.flush()

                if all(not p.is_alive() for p in processes):
                    print("\nAll workers have finished (video sources reached end-of-file). "
                          "Exiting.")
                    break
    except KeyboardInterrupt:
        print("\nShutting down workers...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()