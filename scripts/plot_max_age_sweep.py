from pathlib import Path

import matplotlib.pyplot as plt

MAX_AGE_VALUES = [10, 20, 30, 45, 60]

HOTA = [46.312, 46.726, 46.887, 46.858, 46.472]
IDF1 = [51.827, 52.558, 53.119, 53.193, 52.429]
MOTA = [28.311, 27.677, 27.318, 26.816, 26.577]


def main() -> None:
    output_path = Path("outputs/p5-max-age-sweep.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        MAX_AGE_VALUES,
        HOTA,
        marker="o",
        label="HOTA",
    )
    plt.plot(
        MAX_AGE_VALUES,
        IDF1,
        marker="o",
        label="IDF1",
    )
    plt.plot(
        MAX_AGE_VALUES,
        MOTA,
        marker="o",
        label="MOTA",
    )

    plt.axvline(
        30,
        linestyle="--",
        alpha=0.6,
        label="Selected max_age",
    )

    plt.xlabel("max_age (frames)")
    plt.ylabel("Score")
    plt.title("Track Lifetime Sweep")
    plt.xticks(MAX_AGE_VALUES)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
