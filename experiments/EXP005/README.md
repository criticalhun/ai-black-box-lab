# EXP005 — Friss modellek (seed 3, 4, 5) megerősítő tesztekhez

**Hipotézis:** H001 replikáció (ugyanaz a kritérium, mint az EXP001-ben)
**Státusz:** COMPLETED (2026-10-08) — H001 replikáció SUPPORTED (3/3)

## Cél
Az EXP001-gyel azonos konfiguráció (csak a seedek mások), azonos kód
(experiments/EXP001/train.py). Ezeken a modelleken teszteljük:
- H001 replikáció (itt), H002 replikáció (EXP006), H003 replikáció (EXP007),
- H004–H007 (EXP008), amelyeket a seed 0–2 exploratív vizsgálatából (EXP004)
  fogalmaztunk meg, ezen modellek tanítása ELŐTT.

## Futtatás
    uv run python experiments/EXP001/train.py --config experiments/EXP005/config.yaml --seed {3,4,5}
