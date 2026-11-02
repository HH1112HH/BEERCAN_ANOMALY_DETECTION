# DRAEM branch handoff

Branch: `algorithm/draem`

## Status

Training/checkpoint integration is pending. Do not replace the current runtime with placeholder model files.

## After training

1. Record the training command/config, dataset split, checkpoint checksum, and validation threshold.
2. Implement a DRAEM adapter that follows the shared `AnomalyModel.predict(image) -> AnomalyPrediction` contract in `docs/ANOMALY_MODELS_INTEGRATION.md`.
3. Add known-good/known-defect inference tests and document score/heatmap output shape.
4. Keep the input transform configurable until the planned camera/image-input change is finalized.

Do not commit dataset contents or large checkpoint files directly; choose Git LFS or an artifact registry first.
