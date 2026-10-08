# EXP009 — Direct logit attribution a "=" pozíción (EXPLORATÍV)

**Hipotézis:** nincs (hipotézis-generálás)
**Státusz:** COMPLETED (2026-10-08) — exploratív

## Kérdés
Miért nem kell a kimenethez az MLP-aktivációk egyváltozós része (H007)?
(1) a kiolvasás elnyomja, (2) kioltják egymást, vagy (3) közömbös irányba hatnak?

## Módszer
A "=" pozíció logitjait 12 additív komponensre bontjuk (embed+pos, MLP bias,
valamint attn_out és MLP-aktiváció × 5 2D Fourier-osztály), mindegyiket W_U-n át.
Komponensenként: centrált norma-arány és margin (helyes válasz logitja − átlag).
Kalibráció: tests/test_dla.py (Σ komponens = logit, marginok additívak).

## Fontos
Felfedező modellek: seed 0–2. Az itt született hipotéziseket friss seedeken (6–8) teszteljük.
