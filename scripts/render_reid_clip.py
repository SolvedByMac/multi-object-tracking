from src.tracking.tracker import TrackerConfig
from src.visualisation.render import render_sequence


def main() -> None:
    render_sequence(
        sequence_dir="data/MOT17/train/MOT17-02-FRCNN",
        cache_path="cache/detections/MOT17-02-FRCNN.npz",
        embedding_cache_path="cache/embeddings/MOT17-02-FRCNN.npz",
        output_path="outputs/p4-reid-frames-301-450.mp4",
        start_frame=301,
        end_frame=450,
        tracker_config=TrackerConfig(
            min_iou=0.3,
            n_init=3,
            max_age=30,
            lambda_motion=0.50,
            max_cosine_distance=0.30,
        ),
    )


if __name__ == "__main__":
    main()
