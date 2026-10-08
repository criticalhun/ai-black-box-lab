# EXP004 — Attention, MLP-neuronok és logitok (EXPLORATÍV)

**Hipotézis:** nincs — ez a kísérlet hipotézist GENERÁL, nem tesztel.
**Státusz:** nem futott

## Cél
Megnézni, mit csinál az attention a "=" pozíción, milyen 2D Fourier-szerkezetük van
az MLP-neuronoknak, és mennyire írja le a logitokat a Σ α_k cos(w_k(a+b−c)) képlet.
Minden mérés a végső modellre ÉS a saját init-modelljére (kontroll) is lefut.

## Fontos
Az itt talált mintázatokat NEM tekintjük eredménynek. Ezekből hipotéziseket
(H004+) fogalmazunk, amelyeket előre rögzítve, FRISS seedeken (3, 4, 5) tesztelünk,
hogy elkerüljük a körkörös érvelést (ugyanazon az adaton ötletelni és igazolni).

## Mérőeszköz
`src/fourier2d.py`, kalibrálva: `tests/test_fourier2d.py`.
