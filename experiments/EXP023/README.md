# EXP023 — Friss 2 rétegű seed-batch: elakadás vs alias-margó / frekvenciaszám (H026, H027)

**Státusz:** PREREGISTERED (még nem futott)

12 friss 2 rétegű modell (seed 9–20), split 598, alap-hiperparaméterek (= EXP017 config,
csak `experiment`, `hypothesis` és `seeds` más). Minden modellnél rögzítjük: a végső
frekvencia-spektrumot (57 érték), a K-pályát mind a 41 checkpointon, a korai top-K Jaccardot,
a loss-kiugrásokat (metrics.csv) és az általánosítás lépését.
A referencia-modellek (EXP013 seed 0–2, EXP017 seed 3–5) kiírásra kerülnek, az ítéletbe nem számítanak.

    uv run python experiments/EXP001/train.py --config experiments/EXP023/train/config.yaml --seed {9..20}
    uv run python experiments/EXP023/analyze.py --config experiments/EXP023/analysis.yaml
