from __future__ import annotations

from pathlib import Path

import cv2

from src.detection.cache import load_detection_cache
from src.detection.detector import Detection
from src.reid.cache import load_embedding_cache
from src.tracking.tracker import Tracker, TrackerConfig


def render_sequence(
    sequence_dir: str | Path,
    cache_path: str | Path,
    output_path: str | Path,
    start_frame: int = 1,
    end_frame: int | None = None,
    embedding_cache_path: str | Path | None = None,
    tracker_config: TrackerConfig | None = None,
) -> None:
    sequence_dir = Path(sequence_dir)
    cache_path = Path(cache_path)
    output_path = Path(output_path)

    cache = load_detection_cache(cache_path)

    embedding_cache = None
    if embedding_cache_path is not None:
        embedding_cache = load_embedding_cache(embedding_cache_path)

    first_image = cv2.imread(str(sequence_dir / "img1" / f"{start_frame:06d}.jpg"))

    if first_image is None:
        raise FileNotFoundError(sequence_dir / "img1" / f"{start_frame:06d}.jpg")

    height, width = first_image.shape[:2]
    fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    tracker = Tracker(
        tracker_config
        or TrackerConfig(
            min_iou=0.3,
            n_init=3,
            max_age=30,
        )
    )

    if end_frame is None:
        end_frame = max(cache)

    try:
        for frame_idx in range(start_frame, end_frame + 1):
            image_path = sequence_dir / "img1" / f"{frame_idx:06d}.jpg"

            image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(image_path)

            cached = cache.get(frame_idx)
            detections: list[Detection] = []

            if cached is not None:
                for row in cached:
                    detections.append(
                        Detection(
                            xyxy=row[:4].astype(float),
                            confidence=float(row[4]),
                        )
                    )

            embeddings = None
            if embedding_cache is not None:
                embeddings = embedding_cache.get(frame_idx)

            tracks = tracker.update(
                detections=detections,
                embeddings=embeddings,
                frame_idx=frame_idx,
            )

            for track in tracks:
                x1, y1, x2, y2 = track.to_xyxy().astype(int)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    image,
                    f"ID {track.track_id}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                image,
                f"Frame {frame_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(image)

            if frame_idx % 50 == 0:
                print(f"Rendered frame {frame_idx} ({len(tracks)} confirmed tracks)")

    finally:
        writer.release()

    print(f"Saved video: {output_path}")
