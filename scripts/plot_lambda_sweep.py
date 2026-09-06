from pathlib import Path

import matplotlib.pyplot as plt

LAMBDA_VALUES = [0.00, 0.30, 0.50, 0.70, 0.98]

HOTA = [47.792, 47.405, 46.887, 46.834, 46.803]
IDF1 = [54.874, 53.885, 53.119, 52.984, 52.910]
MOTA = [27.499, 27.312, 27.318, 27.319, 27.319]


def main() -> None:
    output_path = Path("outputs/p5-lambda-sweep.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        LAMBDA_VALUES,
        HOTA,
        marker="o",
        label="HOTA",
    )
    plt.plot(
        LAMBDA_VALUES,
        IDF1,
        marker="o",
        label="IDF1",
    )
    plt.plot(
        LAMBDA_VALUES,
        MOTA,
        marker="o",
        label="MOTA",
    )

    plt.xlabel("Motion weight λ")
    plt.ylabel("Score")
    plt.title("Motion vs Appearance Association Weight")
    plt.xticks(LAMBDA_VALUES)
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
