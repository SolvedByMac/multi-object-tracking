from pathlib import Path

import matplotlib.pyplot as plt

N_INIT_VALUES = [1, 2, 3, 4, 5]

HOTA = [45.233, 46.343, 46.887, 47.504, 47.585]
IDF1 = [49.930, 52.065, 53.119, 54.464, 54.779]
MOTA = [19.577, 24.868, 27.318, 28.815, 30.030]


def main() -> None:
    output_path = Path("outputs/p5-n-init-sweep.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        N_INIT_VALUES,
        HOTA,
        marker="o",
        label="HOTA",
    )
    plt.plot(
        N_INIT_VALUES,
        IDF1,
        marker="o",
        label="IDF1",
    )
    plt.plot(
        N_INIT_VALUES,
        MOTA,
        marker="o",
        label="MOTA",
    )

    plt.axvline(
        3,
        linestyle="--",
        alpha=0.6,
        label="Selected n_init",
    )

    plt.xlabel("n_init (consecutive matches)")
    plt.ylabel("Score")
    plt.title("Track Confirmation Sweep")
    plt.xticks(N_INIT_VALUES)
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
