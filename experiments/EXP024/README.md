# EXP024 — lr × weight decay rács a 2 rétegű instabilitásra (H028)

**Státusz:** COMPLETED — eredmények: JOURNAL (EXP021–EXP024 eredmény-blokk)

Cellák: lr {3e-4, 1e-3, 3e-3} × wd {0, 1.0}, seed {3, 4, 5}. Az alapcella (1e-3, 1.0) az EXP017,
nem futtatjuk újra. Új cellák (mindegyik az EXP017 config másolata, csak `experiment`,
`hypothesis`, `seeds` megjegyzése, `lr`, `weight_decay` más):
lr3e-4_wd1, lr3e-3_wd1, lr3e-4_wd0, lr1e-3_wd0, lr3e-3_wd0 → 15 futás.

    uv run python experiments/EXP001/train.py --config experiments/EXP024/<cella>/config.yaml --seed {3,4,5}
    uv run python experiments/EXP024/analyze.py --config experiments/EXP024/analysis.yaml
