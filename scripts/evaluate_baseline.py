from __future__ import annotations

import sys
from pathlib import Path

TRACK_EVAL_ROOT = Path.home() / "projects" / "TrackEval"
sys.path.insert(0, str(TRACK_EVAL_ROOT))

import trackeval


def main() -> None:
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": True,
            "PRINT_CONFIG": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()

    dataset_config.update(
        {
            "GT_FOLDER": "results/self_eval_gt",
            "TRACKERS_FOLDER": "results/trackers/MOT17-val",
            "TRACKERS_TO_EVAL": ["baseline"],
            "TRACKER_SUB_FOLDER": "data",
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "DO_PREPROC": False,
            "SEQMAP_FILE": "results/seqmaps/MOT17-val.txt",
        }
    )

    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    evaluator = trackeval.Evaluator(eval_config)

    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)

    evaluator.evaluate(
        [dataset],
        metrics,
    )


if __name__ == "__main__":
    main()
