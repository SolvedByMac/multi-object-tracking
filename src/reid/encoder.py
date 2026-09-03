from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchreid


class OSNetEncoder:
    def __init__(
        self,
        weights_path: str | Path,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.batch_size = batch_size

        if not self.weights_path.exists():
            raise FileNotFoundError(f"OSNet weights not found: {self.weights_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            loss="softmax",
            pretrained=False,
        )

        torchreid.utils.load_pretrained_weights(
            self.model,
            str(self.weights_path),
        )

        self.model.to(self.device)
        self.model.eval()

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
        ).view(3, 1, 1)

    @staticmethod
    def _clip_box(
        box: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box[:4]

        x1 = max(0, min(int(np.floor(x1)), frame_width))
        y1 = max(0, min(int(np.floor(y1)), frame_height))
        x2 = max(0, min(int(np.ceil(x2)), frame_width))
        y2 = max(0, min(int(np.ceil(y2)), frame_height))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def _prepare_crop(self, crop: np.ndarray) -> torch.Tensor:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop = cv2.resize(
            crop,
            (128, 256),
            interpolation=cv2.INTER_LINEAR,
        )

        tensor = torch.from_numpy(crop).float()
        tensor = tensor.permute(2, 0, 1) / 255.0

        tensor = (tensor - self.mean) / self.std

        return tensor

    def encode(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:

        boxes = np.asarray(boxes, dtype=np.float32)

        if boxes.ndim != 2 or boxes.shape[1] < 4:
            raise ValueError("boxes must have shape (N, >=4)")

        num_boxes = len(boxes)

        if num_boxes == 0:
            return np.empty((0, 512), dtype=np.float32)

        frame_height, frame_width = frame.shape[:2]

        valid_indices: list[int] = []
        crops: list[torch.Tensor] = []

        for index, box in enumerate(boxes):
            clipped = self._clip_box(
                box,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            if clipped is None:
                continue

            x1, y1, x2, y2 = clipped
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crops.append(self._prepare_crop(crop))
            valid_indices.append(index)

        embeddings = np.zeros(
            (num_boxes, 512),
            dtype=np.float32,
        )

        if not crops:
            return embeddings

        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                end = start + self.batch_size

                batch = torch.stack(crops[start:end])
                batch = batch.to(self.device)

                features = self.model(batch)
                features = F.normalize(features, p=2, dim=1)

                features_np = features.cpu().numpy().astype(np.float32)

                indices = valid_indices[start:end]

                for output_index, feature in zip(
                    indices,
                    features_np,
                    strict=True,
                ):
                    embeddings[output_index] = feature

        return embeddings
