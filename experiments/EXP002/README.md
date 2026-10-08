# EXP002 — Fourier-szerkezet az EXP001 embeddingjeiben

**Hipotézis:** H002
**Státusz:** COMPLETED (2026-10-08) — H002 SUPPORTED

## Cél
Megvizsgálni, hogy a betanított modellek számembeddingjei Fourier-bázisban
koncentráltak-e (néhány frekvencia viszi az energia nagy részét).
Ez KORRELÁCIÓS/strukturális vizsgálat: azt nézi, mi VAN a súlyokban,
nem azt, hogy a modell használja-e (az a H003 kauzális tesztje lesz).

## Bemenet
Az EXP001 checkpointjai (`experiments/EXP001/results/seed{N}/checkpoints/`).
A checkpointok nincsenek gitben; az EXP001 determinisztikus, így újragenerálhatók
(ellenőrzés: `summary.json` → `final_state_sha256`).

## Mérőeszköz
`src/fourier.py`, kalibrálva: `tests/test_fourier.py`
(ortonormalitás, Parseval, véletlen referencia ≈ 0,16, ismert {5,17} szerkezet visszanyerése).

## Amit az eredmény NEM bizonyít
Ha az energia koncentrált, az nem bizonyítja, hogy a modell ezekre a
frekvenciákra támaszkodik. Ehhez beavatkozás kell (ablation → H003).
