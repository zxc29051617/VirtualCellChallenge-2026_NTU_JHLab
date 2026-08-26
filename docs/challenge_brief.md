# Virtual Cell Challenge 2026 Brief

## What the challenge asks

- Predict CRISPRi knockdown responses from unperturbed control cells.
- Validation uses cell contexts `A`, `B`, and `C`.
- Final test uses held-out contexts `D`, `E`, and `F`.
- Each context has 300 perturbations and 18,400 control cells in the validation bundle.

## Official submission constraints

- One submission file must cover all three contexts.
- Use the official context labels from the bundle you are predicting for.
- Predict exactly the official perturbation panel.
- Use raw counts, not log-normalized values.
- Do not include non-targeting control cells in the final submission.
- The leaderboard limits each team to two scored submissions per day.

## Timing

- Validation release is live.
- Final test data is released on 2026-10-22.
- Final submission deadline is 2026-11-05.

## Evaluation

- The official evaluator uses six scored metrics plus an overall unweighted mean.
- The broad metric families are discrimination, expression accuracy, and DE-based scores.
- The official `vcc` CLI validates the submission format locally before upload.

## Notes from this workspace

- `dataset/` contains the validation control bundle for contexts `A`, `B`, and `C`.
- `dataset/gene_names.csv` has a header row locally; the official tooling expects a headerless gene list, so the helper code strips it when needed.
- The first baseline is a context-mean predictor, sparsified to keep the official prep tool within memory.

