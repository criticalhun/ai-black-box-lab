# EXP003 — Kulcsfrekvenciák ablációja az embeddingben

**Hipotézis:** H003
**Státusz:** COMPLETED (2026-10-08) — H003 SUPPORTED

## Cél
Kauzális teszt: a modell valóban a W_E kulcsfrekvenciáira támaszkodik-e?
Szükségesség (eltávolítás), elégségesség (csak ezek megtartása), véletlen kontrollal.

## Amit az eredmény NEM bizonyít
Nem mutatja meg, hogyan számol az attention és az MLP; csak azt, hogy az
embedding melyik része hordozza a feladathoz szükséges információt.
