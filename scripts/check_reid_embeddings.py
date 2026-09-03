from pathlib import Path

import cv2
import numpy as np

from src.data.formats import tlwh_to_xyxy
from src.data.mot_dataset import load_ground_truth
from src.reid.encoder import OSNetEncoder

SEQUENCE_DIR = Path("data/MOT17/train/MOT17-02-FRCNN")

WEIGHTS = Path(
    "weights/reid/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_"
    "b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)

NUM_PAIRS = 20
MIN_FRAME_GAP = 5
MAX_FRAME_GAP = 20
MIN_VISIBILITY = 0.5
MIN_HEIGHT = 80.0


def load_frame(frame_idx: int) -> np.ndarray:
    path = SEQUENCE_DIR / "img1" / f"{frame_idx:06d}.jpg"

    frame = cv2.imread(str(path))

    if frame is None:
        raise RuntimeError(f"Could not load frame: {path}")

    return frame


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Embeddings are already L2-normalized.
    return float(1.0 - np.dot(a, b))


def valid_gt_rows(gt: np.ndarray) -> np.ndarray:

    mark = gt[:, 6]
    class_id = gt[:, 7]
    visibility = gt[:, 8]
    height = gt[:, 5]

    mask = (
        (mark == 1)
        & (class_id == 1)
        & (visibility >= MIN_VISIBILITY)
        & (height >= MIN_HEIGHT)
    )

    return gt[mask]


def find_same_id_pairs(gt: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    identities = np.unique(gt[:, 1].astype(int))

    for identity in identities:
        rows = gt[gt[:, 1] == identity]
        rows = rows[np.argsort(rows[:, 0])]

        for i, row_a in enumerate(rows):
            frame_a = int(row_a[0])

            for row_b in rows[i + 1 :]:
                frame_b = int(row_b[0])
                gap = frame_b - frame_a

                if gap < MIN_FRAME_GAP:
                    continue

                if gap > MAX_FRAME_GAP:
                    break

                pairs.append((row_a, row_b))
                break

            if len(pairs) >= NUM_PAIRS:
                return pairs

    return pairs


def find_different_id_pairs(
    gt: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    frames = np.unique(gt[:, 0].astype(int))

    for frame_idx in frames:
        rows = gt[gt[:, 0] == frame_idx]

        if len(rows) < 2:
            continue

        for i in range(len(rows) - 1):
            row_a = rows[i]

            for j in range(i + 1, len(rows)):
                row_b = rows[j]

                if int(row_a[1]) == int(row_b[1]):
                    continue

                pairs.append((row_a, row_b))

                if len(pairs) >= NUM_PAIRS:
                    return pairs

    return pairs


def embed_gt_row(
    encoder: OSNetEncoder,
    row: np.ndarray,
) -> np.ndarray:
    frame_idx = int(row[0])

    frame = load_frame(frame_idx)

    tlwh = row[2:6]
    xyxy = tlwh_to_xyxy(tlwh)

    embedding = encoder.encode(
        frame,
        np.asarray([xyxy], dtype=np.float32),
    )[0]

    return embedding


def main() -> None:
    gt = load_ground_truth(SEQUENCE_DIR)
    gt = valid_gt_rows(gt)

    print(f"Usable GT rows: {len(gt)}")

    same_pairs = find_same_id_pairs(gt)
    different_pairs = find_different_id_pairs(gt)

    if len(same_pairs) < NUM_PAIRS:
        raise RuntimeError(f"Only found {len(same_pairs)} same-ID pairs")

    if len(different_pairs) < NUM_PAIRS:
        raise RuntimeError(f"Only found {len(different_pairs)} different-ID pairs")

    encoder = OSNetEncoder(WEIGHTS)

    same_distances = []
    different_distances = []

    print("\nSame-ID pairs:")

    for index, (row_a, row_b) in enumerate(same_pairs, start=1):
        emb_a = embed_gt_row(encoder, row_a)
        emb_b = embed_gt_row(encoder, row_b)

        distance = cosine_distance(emb_a, emb_b)
        same_distances.append(distance)

        print(
            f"{index:02d}: "
            f"ID {int(row_a[1])} | "
            f"frames {int(row_a[0])} → {int(row_b[0])} | "
            f"distance={distance:.4f}"
        )

    print("\nDifferent-ID pairs:")

    for index, (row_a, row_b) in enumerate(
        different_pairs,
        start=1,
    ):
        emb_a = embed_gt_row(encoder, row_a)
        emb_b = embed_gt_row(encoder, row_b)

        distance = cosine_distance(emb_a, emb_b)
        different_distances.append(distance)

        print(
            f"{index:02d}: "
            f"IDs {int(row_a[1])} vs {int(row_b[1])} | "
            f"frame {int(row_a[0])} | "
            f"distance={distance:.4f}"
        )

    same_mean = float(np.mean(same_distances))
    different_mean = float(np.mean(different_distances))

    print("\n--- Summary ---")
    print(f"Same-ID mean cosine distance:      {same_mean:.4f}")
    print(f"Different-ID mean cosine distance: {different_mean:.4f}")
    print(f"Separation:                        {different_mean - same_mean:.4f}")

    if same_mean >= different_mean:
        raise RuntimeError(
            "Re-ID sanity check FAILED: same-ID embeddings are not closer on average."
        )

    print("\nPASSED: same-ID embeddings are closer on average.")


if __name__ == "__main__":
    main()
