# AI Black Box Lab

> Don't ask the model why. Look inside.

Long-term, hands-on mechanistic interpretability research: train small,
fully inspectable neural networks on controlled tasks, then reconstruct
*from their internals* why they produce a given output — and test those
claims with causal interventions.

## Principles
- Observation, interpretation and hypothesis are kept separate.
- Activation alone does not prove mechanism; causal tests do.
- Hypotheses and success criteria are committed *before* experiments run.
- Old experiments are never silently modified — changes get a new EXP id.
- Negative results (HYPOTHESIS REJECTED) are documented too.

## Layout
- `JOURNAL.md` — chronological research log
- `hypotheses/Hxxx.md` — one file per hypothesis, with status
- `experiments/EXPxxx/` — config, code, README, results per experiment
- `src/` — shared code (models, data, hooks); `tests/` — sanity checks
- `analysis/`, `figures/`, `notes/`

## Reproduce
Requires `uv` and an NVIDIA GPU with a driver supporting CUDA >= 13.0.

    uv python install 3.12
    uv sync
    uv run python experiments/EXP001/train.py --config experiments/EXP001/config.yaml --seed 0

Model checkpoints (`*.pt`) are not tracked in git; they are regenerated
from configs + seeds.
