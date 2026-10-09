# EXP019 — Mit készít elő a 0. réteg? (EXPLORATÍV)

**Hipotézis:** nincs (hipotézis-generálás)
**Státusz:** nem futott

Felfedező modellek: EXP013 seed 0–2. Mérések: attention rétegenként (honnan hova figyel),
az egyes komponensek kimenetének 2D Fourier-összetétele, és az MLP0 egyváltozós energiájának
helye (nagy frekvenciák / 2.–3. felharmonikusok / egyéb), a bemenetéhez és az inithez képest.
Kalibráció: tests/test_between_layers.py.
