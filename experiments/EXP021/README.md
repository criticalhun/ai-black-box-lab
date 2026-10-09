# EXP021 — Split-variancia: H001-replika új adatfelosztásokon (H023, H024)

**Státusz:** PREREGISTERED (még nem futott)

Keresztezett terv, 1 rétegű modell: data_seed {598, 101, 202} × modell-seed {0, 1, 2}.
A 598-as sor az EXP001 meglévő futásai; új futás csak a 101-es és 202-es felosztásra kell (6 db).
Az új configok az EXP001 config másolatai; csak az `experiment`, a `hypothesis` és a `data_seed` sor más.

Tanítás (kód: experiments/EXP001/train.py, változatlan):

    uv run python experiments/EXP001/train.py --config experiments/EXP021/d101/config.yaml --seed {0,1,2}
    uv run python experiments/EXP001/train.py --config experiments/EXP021/d202/config.yaml --seed {0,1,2}

Elemzés (előre commitolva):

    uv run python experiments/EXP021/analyze.py --config experiments/EXP021/analysis.yaml
