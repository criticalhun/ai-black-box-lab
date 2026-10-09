# EXP016 — Hol számol a 2 rétegű modell? (EXPLORATÍV)

**Hipotézis:** nincs (hipotézis-generálás)
**Státusz:** nem futott

Felfedező modellek: EXP013 seed 0–2. Három MLP-hely: (réteg 0, b pozíció), (réteg 0, "="),
(réteg 1, "="). Mérések: neuron-szerkezet (2D Fourier), közvetlen logit-attribúció rétegenként,
átlag-abláció (attn/mlp rétegenként), szorzat- és egyváltozós tagok kivetítése helyenként.
Kalibráció: tests/test_layers.py. Megerősítés később friss 2 rétegű seedeken.
