# EXP010 — MLP-neuronok logit-hozzájárulásainak koherenciája (EXPLORATÍV)

**Hipotézis:** nincs (hipotézis-generálás)
**Státusz:** COMPLETED (2026-10-08) — exploratív

## Kérdés
Az egyváltozós tagok a neuronok közötti kioltás miatt tűnnek-e el a logitokból,
miközben a szorzat-tagok erősítik egymást?
Mérőszám: kappa = ||Σ_n hozzájárulás_n||² / Σ_n ||hozzájárulás_n||²
(kalibráció: tests/test_coherence.py). Felfedező modellek: seed 0–2, + init kontroll.
