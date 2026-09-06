# Multi-Camera Face Search (Live RTSP)

Upload a reference photo of a person, and this searches for that person
across all 4 (or however many) of your live RTSP camera streams in
real time, alerting with camera ID, timestamp, and a snapshot whenever
there's a match.

**Tested and working**: this uses [InsightFace](https://github.com/deepinsight/insightface)'s
`buffalo_l` model (real face detection + 512-dim face embeddings, the same
class of model used in commercial face-search products). In this environment,
enrolling a reference photo and comparing it against a matching vs. non-matching
face scored **0.998 similarity for the same person** and **0.03 for a different
one**, against a match threshold of 0.45 — confirming the detection and
matching pipeline works correctly before you point it at real cameras.

## Before you run this: legal requirements

This tool identifies and tracks a specific named individual across physical
space. That's meaningfully different from generic object detection, and most
jurisdictions regulate it specifically:

- **Only use this on cameras/property you own or are authorized to monitor.**
- **Post visible notice** that cameras with facial recognition are in use —
  required in most US states, the EU (GDPR), and elsewhere.
- **Biometric-specific laws**: Illinois (BIPA), Texas (CUBI), Washington, and
  the EU (GDPR Art. 9) specifically regulate storing/matching face embeddings
  and generally require documented consent or a narrow legal basis (e.g.
  security of the premises) — check your jurisdiction before deploying.
- **Don't use this to track someone without their knowledge in a context
  they'd reasonably expect privacy**, or on cameras you don't control. Aside
  from being illegal in most places, that's stalking.
- Legitimate uses this is built for: security teams monitoring their own
  building/store with proper notice, finding a lost person (e.g. a child or
  elderly relative) at a large venue you operate, or authorized access
  control.

## How it works

```
reference_photos/your_photo.jpg
        │
        ▼
enroll_reference.py  ──►  512-dim face embedding ("faceprint") saved to disk
        │
        ▼
main.py launches ONE process per camera (stream_worker.py)
        │
        ├── Camera 1 (RTSP) ─► detect faces ─► compare to faceprint ─► alert if match
        ├── Camera 2 (RTSP) ─► detect faces ─► compare to faceprint ─► alert if match
        ├── Camera 3 (RTSP) ─► detect faces ─► compare to faceprint ─► alert if match
        └── Camera 4 (RTSP) ─► detect faces ─► compare to faceprint ─► alert if match
                        │
                        ▼
        All matches merged into logs/alerts.jsonl (+ snapshot images)
```

Each camera runs as its own OS process, so if one camera's stream drops or
stalls, the other three keep running unaffected. `stream_worker.py` also
auto-reconnects with backoff if an RTSP connection drops.

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads the face model weights automatically (~280MB, from
InsightFace's GitHub releases) — this happens once and is cached at
`~/.insightface/models/`.

### Step 1 — Enroll the reference photo

```bash
python src/enroll_reference.py --photo reference_photos/your_photo.jpg
```

Use a clear, front-facing, well-lit, single-person photo for best accuracy.
This creates `reference_photos/target.faceprint` — the encoded vector every
camera will compare against.

### Step 2 — Configure your 4 sources (live cameras OR video files)

Edit `cameras.yaml`. Each entry's `source` can be **either** a live RTSP URL
**or** a local video file path — auto-detected, and freely mixable:

```yaml
# Live cameras:
cameras:
  - id: "entrance"
    source: "rtsp://username:password@192.168.1.10:554/stream1"
  - id: "checkout_1"
    source: "rtsp://username:password@192.168.1.11:554/stream1"
  - id: "checkout_2"
    source: "rtsp://username:password@192.168.1.12:554/stream1"
  - id: "exit"
    source: "rtsp://username:password@192.168.1.13:554/stream1"
```

```yaml
# Local video files (e.g. a demo with 4 pre-recorded clips):
loop_video: false   # set true to loop each clip instead of stopping at its end
cameras:
  - id: "angle_front"
    source: "clips/person_front.mp4"
  - id: "angle_left"
    source: "clips/person_left.mp4"
  - id: "angle_right"
    source: "clips/person_right.mp4"
  - id: "angle_back"
    source: "clips/person_back.mp4"
```

The RTSP URL format varies by camera brand — check your camera's admin
page/manual. Most IP cameras (Hikvision, Dahua, Reolink, Amcrest, etc.)
expose it under a "RTSP settings" or "network" menu.

**Behavior differences between the two source types:**
- **Live RTSP**: runs forever, auto-reconnects with backoff if the
  connection drops. `main.py` keeps running until you Ctrl+C it.
- **Local video file**: plays through once and the worker exits cleanly at
  end-of-file (unless `loop_video: true`). Once every source has finished,
  `main.py` itself exits automatically — no need to Ctrl+C.

Tested: running the pipeline against 4 local video clips of the same
enrolled face correctly matched all 4 (similarity 0.96-0.98), and the
process exited cleanly on its own once all 4 clips finished.

### Step 3 — Run

```bash
python src/main.py --config cameras.yaml --faceprint reference_photos/target.faceprint
```

You'll see console output like:
```
Launched worker for camera 'entrance' (PID 12345)
Launched worker for camera 'checkout_1' (PID 12346)
...
[MATCH] Camera 'checkout_1' at 14:32:07 (similarity=0.812)
```

Matches are appended to `logs/alerts.jsonl` with camera ID, timestamp,
similarity score, and a saved snapshot path.

## Tuning

- **`--threshold`** (default 0.45): raise it (e.g. 0.55-0.6) to reduce false
  matches at the cost of possibly missing some real matches; lower it to
  catch more matches at the cost of more false positives. Tune this against
  a few real clips of your target person before relying on it operationally.
- **`sample_every_n_frames`** in `stream_worker.py` (default 5): higher values
  reduce CPU load by checking fewer frames per second; lower values catch
  someone who passes through frame quickly but cost more compute.
- **CPU vs GPU**: this runs on CPU by default (`CPUExecutionProvider`). For
  4+ simultaneous streams at good frame rates, a GPU makes a big difference —
  install `onnxruntime-gpu` instead of `onnxruntime` and change the provider
  in `face_engine.py` to `CUDAExecutionProvider` if you have an NVIDIA GPU.

## Files

```
multi-cam-face-search/
├── src/
│   ├── face_engine.py       # Shared face detection + embedding model wrapper
│   ├── enroll_reference.py  # Encodes your reference photo into a faceprint
│   ├── stream_worker.py     # Per-camera RTSP loop: detect, compare, alert
│   └── main.py              # Launches all camera workers, merges alerts
├── cameras.yaml              # Your camera list — edit this with real RTSP URLs
├── reference_photos/         # Put your reference photo here
├── logs/
│   ├── alerts.jsonl          # Match log (created on first run)
│   └── snapshots/            # Saved frame images for each match
└── requirements.txt
```

## What this prototype does NOT include (production considerations)

- **No dashboard/UI** — alerts currently just append to a JSONL log file.
  For real use you'd want a simple web view showing recent matches with
  snapshots, or push alerts to Slack/email/SMS.
- **No database** — for long-running deployments, write alerts to a real
  database (Postgres/SQLite) instead of a flat file.
- **No de-duplication** — if the person stays in frame for 30 seconds, you'll
  get repeated match alerts. Add a cooldown (e.g. don't re-alert for the same
  camera within N seconds of the last match) if that's noisy for your use case.
- **Single reference photo per run** — to search for multiple people
  simultaneously, extend `enroll_reference.py` to store multiple named
  faceprints and check incoming faces against all of them.
