# EXP001 — 1 rétegű Transformer, moduláris összeadás

**Hipotézis:** H001
**Státusz:** COMPLETED (2026-10-08) — H001 SUPPORTED

## Cél
Létrehozni a vizsgálat tárgyát: egy kis Transformert, amely a feladatot
generalizálja, nem csak memorizálja. Ez a kísérlet NEM értelmez semmit
a belső működésről.

## Miért moduláris összeadás?
Ismert, publikált mechanizmus van rá (Nanda et al., 2023, "Progress measures
for grokking via mechanistic interpretability"). Ez kalibrációs pont: ha a
későbbi elemző eszközeink nem találják meg az ismert mechanizmust,
az eszközeinkben van a hiba. A sima (nem moduláris) összeadás: egy későbbi EXP (az EXP002 a modell elemzése lett).

## Feladat
Bemenet: `[a, b, =]`, cél: `(a + b) mod 113`. Minden szám egyetlen token.
Teljes tér: 113² = 12 769 pár → 30% train (fix felosztás), 70% test.

## Futtatás
Minden EXP001-futás ugyanazon a gépen (fedora, RTX 4060), CUDA-n,
determinisztikus módban. CPU és GPU eredményei bitre nem összevethetők.

## Mérések
- train/test loss és accuracy `log_every` lépésenként → `results/seed{N}/metrics.csv`
- checkpointok → `results/seed{N}/checkpoints/` (nincs gitben)
- determinizmus-ellenőrzés: seed 0 kétszer

## Amit az eredmény NEM bizonyít
A magas teszt-pontosság csak azt mutatja, hogy van mit vizsgálni.
A mechanizmusra vonatkozó állítások későbbi hipotézisek (H002+) tárgyai.

## Eredmények (2026-10-08)
Kód: commit f995dc3. Preregisztráció: commit a188156.

| futás | train acc ≥ 0.99 | test acc ≥ 0.99 | végső test acc | végső test loss | súly-hash |
|---|---|---|---|---|---|
| seed0 | 200 | 10400 | 1.0000 | 7.23e-07 | 8085b4b645d6 |
| seed1 | 200 | 8500 | 1.0000 | 1.64e-07 | 7845fe4fc865 |
| seed2 | 200 | 10900 | 1.0000 | 1.74e-07 | e35cf6e46d3d |
| seed0_rerun | 200 | 10400 | 1.0000 | 7.23e-07 | 8085b4b645d6 |

- Determinizmus: seed0 vs seed0_rerun → metrics.csv bit-azonos, végső súly-hash azonos.
- Futásidő: ~315 s / futás (~7.9 ms / lépés), RTX 4060.
