# DRAEM and SK-RD4AD integration plan

This document is the handoff for integrating the two anomaly models after training. It deliberately does not assume a checkpoint format, input resolution, score calibration, or final preprocessing until the trained artifacts and the revised image-input flow are available.

## Git branches

- `algorithm/draem` — DRAEM training/inference adapter and its evaluation.
- `algorithm/sk-rd4ad` — SK-RD4AD training/inference adapter and its evaluation.

The branches share the React/Vite frontend and API contract. Keep model-specific code and tests isolated in each branch; merge an adapter to the common branch only after both return the same result shape.

## Important: image input will change

**TODO before integrating either checkpoint:** the image input logic is expected to change. Do not hard-code the current camera crop/ROI/resize behavior into either model adapter. First confirm the future source, color order, frame size, crop/ROI, normalization, and trigger-to-frame selection. Keep acquisition and preprocessing as separate components so the camera/ROI logic can change without rewriting DRAEM or SK-RD4AD inference.

Current path to review when that decision is made:

1. `CameraService` acquires RTSP frames and provides `get_frame_near()`.
2. `YOLODetector` finds a can and produces an ROI.
3. The anomaly model receives the selected image/ROI.
4. `Classifier` combines the anomaly result into `GOOD`/`NG`.

Confirm which of these steps remains before implementing model-specific transforms.

## Shared inference contract

Each algorithm should implement the same backend adapter contract (exact class names may be adjusted during integration):

```python
class AnomalyModel:
    def predict(self, image) -> AnomalyPrediction: ...

@dataclass
class AnomalyPrediction:
    score: float
    is_anomaly: bool
    threshold: float
    heatmap: object | None = None
    mask: object | None = None
    metadata: dict = field(default_factory=dict)
```

The adapter owns model loading and model-specific preprocessing/postprocessing. It must not open the camera, choose the product trigger frame, or control the ESP. Return a score and optional spatial maps in the same format so the API, history, and frontend stay algorithm-agnostic.

## Per-branch checklist

### DRAEM (`algorithm/draem`)

- Add the trained checkpoint path and architecture/config used for training.
- Implement DRAEM loading and inference in an isolated adapter.
- Document expected input range/color order and score/heatmap semantics from the actual training code.
- Add tests with at least one known-good and one known-anomalous sample.

### SK-RD4AD (`algorithm/sk-rd4ad`)

- Add the trained checkpoint path and architecture/config used for training.
- Implement SK-RD4AD loading and inference in an isolated adapter.
- Document expected input range/color order and score/heatmap semantics from the actual training code.
- Add tests with at least one known-good and one known-anomalous sample.

## Config and artifacts

After training, add a model selector such as `anomaly.algorithm: draem | sk-rd4ad` and a model-specific checkpoint/config path. Do not guess thresholds: calibrate each model on a held-out validation set and record the metric, threshold, dataset split, and date in the branch documentation.

Do not commit datasets or large checkpoints as ordinary Git files. Choose Git LFS or an artifact/model registry before uploading trained weights; keep local paths and secrets out of tracked `config.yaml` and frontend environment variables.

## Validation before deployment

1. Confirm the final image-input/ROI contract.
2. Load a checkpoint on CPU and the target GPU; check output dimensions and finite scores.
3. Evaluate GOOD/NG precision/recall and false-reject rate on a held-out set.
4. Verify heatmap alignment to the image/ROI shown by the UI.
5. Run the shared API contract and confirm `GOOD`/`NG` still maps to the ESP protocol.
6. Record checkpoint checksum, threshold, runtime, and known limitations in the algorithm branch.
