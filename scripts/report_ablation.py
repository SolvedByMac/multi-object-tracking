from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    tracker: str
    label: str
    use_mahalanobis: bool
    use_appearance: bool
    use_low_confidence_recovery: bool
    high_confidence: float
    low_confidence: float | None
    min_iou: float = 0.3
    n_init: int = 3
    max_age: int = 30
    lambda_motion: float = 0.50
    max_cosine_distance: float = 0.30


CONFIGS = [
    ExperimentConfig(
        tracker="iou_only",
        label="IoU only (SORT-equiv)",
        use_mahalanobis=False,
        use_appearance=False,
        use_low_confidence_recovery=False,
        high_confidence=0.10,
        low_confidence=None,
    ),
    ExperimentConfig(
        tracker="mahalanobis",
        label="+ Mahalanobis gating",
        use_mahalanobis=True,
        use_appearance=False,
        use_low_confidence_recovery=False,
        high_confidence=0.10,
        low_confidence=None,
    ),
    ExperimentConfig(
        tracker="appearance",
        label="+ appearance (DeepSORT-equiv)",
        use_mahalanobis=True,
        use_appearance=True,
        use_low_confidence_recovery=False,
        high_confidence=0.10,
        low_confidence=None,
    ),
    ExperimentConfig(
        tracker="final",
        label="+ low-conf recovery (final)",
        use_mahalanobis=True,
        use_appearance=True,
        use_low_confidence_recovery=True,
        high_confidence=0.50,
        low_confidence=0.10,
    ),
]

METRICS = [
    "MOTA",
    "IDF1",
    "HOTA",
    "DetA",
    "AssA",
    "IDSW",
    "Frag",
]


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_detailed_results(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def extract_metrics(row: dict[str, str]) -> dict[str, float | int]:
    return {
        "MOTA": float(row["MOTA"]) * 100.0,
        "IDF1": float(row["IDF1"]) * 100.0,
        "HOTA": float(row["HOTA___AUC"]) * 100.0,
        "DetA": float(row["DetA___AUC"]) * 100.0,
        "AssA": float(row["AssA___AUC"]) * 100.0,
        "IDSW": int(float(row["IDSW"])),
        "Frag": int(float(row["Frag"])),
    }


def load_experiment(
    config: ExperimentConfig,
    tracker_root: Path,
) -> tuple[
    dict[str, float | int],
    dict[str, dict[str, float | int]],
]:
    path = tracker_root / config.tracker / "pedestrian_detailed.csv"

    rows = load_detailed_results(path)

    combined: dict[str, float | int] | None = None
    sequences: dict[str, dict[str, float | int]] = {}

    for row in rows:
        sequence_name = row["seq"]
        metrics = extract_metrics(row)

        if sequence_name == "COMBINED":
            combined = metrics
        else:
            sequences[sequence_name] = metrics

    if combined is None:
        raise ValueError(f"No COMBINED row found in {path}")

    return combined, sequences


def write_aggregate_csv(
    path: Path,
    results: list[
        tuple[
            ExperimentConfig,
            dict[str, float | int],
        ]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Config",
                *METRICS,
            ]
        )

        for config, metrics in results:
            writer.writerow(
                [
                    config.label,
                    *[metrics[metric] for metric in METRICS],
                ]
            )


def write_per_sequence_csv(
    path: Path,
    results: list[
        tuple[
            ExperimentConfig,
            dict[str, dict[str, float | int]],
        ]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Config",
                "Sequence",
                *METRICS,
            ]
        )

        for config, sequences in results:
            for sequence_name, metrics in sequences.items():
                writer.writerow(
                    [
                        config.label,
                        sequence_name,
                        *[metrics[metric] for metric in METRICS],
                    ]
                )


def format_metric(
    metric: str,
    value: float,
) -> str:
    if metric in {"IDSW", "Frag"}:
        return str(int(value))

    return f"{float(value):.3f}"


def write_markdown(
    path: Path,
    results: list[
        tuple[
            ExperimentConfig,
            dict[str, float | int],
        ]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Phase 5 Cumulative Ablation",
        "",
        "| Config | MOTA | IDF1 | HOTA | DetA | AssA | IDSW | Frag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for config, metrics in results:
        values = [
            format_metric(
                metric,
                metrics[metric],
            )
            for metric in METRICS
        ]

        lines.append(
            "| "
            + " | ".join(
                [
                    config.label,
                    *values,
                ]
            )
            + " |"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def append_jsonl(
    path: Path,
    results: list[
        tuple[
            ExperimentConfig,
            dict[str, float | int],
            dict[str, dict[str, float | int]],
        ]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(UTC).isoformat()

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        for config, aggregate, sequences in results:
            record = {
                "timestamp": timestamp,
                "config_hash": config_hash(config),
                "config": asdict(config),
                "aggregate": aggregate,
                "per_sequence": sequences,
            }

            file.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
                + "\n"
            )


def print_table(
    results: list[
        tuple[
            ExperimentConfig,
            dict[str, float | int],
        ]
    ],
) -> None:
    headers = [
        "Config",
        *METRICS,
    ]

    rows = []

    for config, metrics in results:
        rows.append(
            [
                config.label,
                *[
                    format_metric(
                        metric,
                        metrics[metric],
                    )
                    for metric in METRICS
                ],
            ]
        )

    widths = [
        max(
            len(headers[column]),
            max(len(row[column]) for row in rows),
        )
        for column in range(len(headers))
    ]

    header = " | ".join(
        headers[column].ljust(widths[column]) for column in range(len(headers))
    )

    separator = "-+-".join("-" * width for width in widths)

    print(header)
    print(separator)

    for row in rows:
        print(
            " | ".join(row[column].ljust(widths[column]) for column in range(len(row)))
        )


def main() -> None:
    tracker_root = Path("results/trackers/MOT17-val")

    output_root = Path("outputs")

    aggregate_results: list[
        tuple[
            ExperimentConfig,
            dict[str, float | int],
        ]
    ] = []

    sequence_results: list[
        tuple[
            ExperimentConfig,
            dict[str, dict[str, float | int]],
        ]
    ] = []

    log_results: list[
        tuple[
            ExperimentConfig,
            dict[str, float | int],
            dict[str, dict[str, float | int]],
        ]
    ] = []

    for config in CONFIGS:
        aggregate, sequences = load_experiment(
            config,
            tracker_root,
        )

        aggregate_results.append(
            (
                config,
                aggregate,
            )
        )

        sequence_results.append(
            (
                config,
                sequences,
            )
        )

        log_results.append(
            (
                config,
                aggregate,
                sequences,
            )
        )

    write_aggregate_csv(
        output_root / "p5-ablation-results.csv",
        aggregate_results,
    )

    write_per_sequence_csv(
        output_root / "p5-ablation-per-sequence.csv",
        sequence_results,
    )

    write_markdown(
        output_root / "p5-ablation-results.md",
        aggregate_results,
    )

    append_jsonl(
        output_root / "p5-ablation-runs.jsonl",
        log_results,
    )

    print_table(aggregate_results)

    print()
    print("Saved outputs/p5-ablation-results.csv")
    print("Saved outputs/p5-ablation-per-sequence.csv")
    print("Saved outputs/p5-ablation-results.md")
    print("Appended outputs/p5-ablation-runs.jsonl")


if __name__ == "__main__":
    main()
