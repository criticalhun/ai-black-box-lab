# Eredmények — 1. fázis: moduláris összeadás (2026-10-08)

## Kérdés
Hogyan oldja meg egy 1 rétegű Transformer a `(a + b) mod 113` feladatot — a modell
belső működéséből rekonstruálva, nem a modell "magyarázatából"?

## Módszertan
- Minden hipotézis és küszöb a tesztelés ELŐTT commitolva (hypotheses/Hxxx.md);
  a H008–H009-nél az elemző kód is.
- Felfedezés a seed 0–2 modelleken; megerősítés olyan FRISS modelleken (seed 3–5, 6–8),
  amelyek a hipotézis rögzítése után születtek.
- Minden mérőeszköz ismert válaszú esetekre kalibrálva (tests/).
- Determinisztikus GPU-tanítás (bit-azonos újrafuttatás igazolva), checkpoint-hash
  ellenőrzés minden elemzésben.

## Megerősített állítások
| # | Állítás | Típus | Bizonyíték |
|---|---|---|---|
| H001 | A modell általánosít (≥ 99% teszt-pontosság), késleltetve ("grokking") | viselkedés | 9/9 seed |
| H002 | A számembeddingek energiájának > 98%-a ≤ 8 Fourier-frekvencián | strukturális | 6/6 seed |
| H003 | Ezek a frekvenciák szükségesek és (embedding-szinten) elégségesek | **kauzális** | 6/6 seed |
| H005 | Minden MLP-neuron egy nagy frekvenciára hangolt; szorzat-tagok ~20% (init: 0.1%) | strukturális | seed 3–5 |
| H007 | A szorzat-tagok kellenek; az egyváltozós tagok (~76% variancia) nem | **kauzális** | seed 3–5 |
| H008 | Kiolvasáskor az egyváltozós tagok kioltódnak (κ ≤ 0.04), a szorzat-tagok erősítik egymást (κ ≥ 20) | strukturális | seed 6–8 |
| H009 | A helyes válasz logit-előnyének ~96%-át az MLP szorzat-tagjai adják | attribúció | seed 6–8 |

## Részleges
| # | Állítás | Eredmény |
|---|---|---|
| H004 | Minden fej ~50–50% arányban figyel a-ra és b-re, "="-re ≤ 0.15 | 2/3 seed |
| H006 | Logitok ≈ Σ α_k cos(w_k(a+b−c)), R² ≥ 0.90 | 2/3 seed (R² 0.88–0.99) |

## A mechanizmus (a fenti állításokból)
1. **Embedding:** minden szám néhány (3–5) frekvencián, körön elhelyezett pontként kódolva.
2. **Attention:** a "=" pozíció összegyűjti a és b információját.
3. **MLP:** a frekvenciánként klasztereződő neuronok ReLU-val szorzat-tagokat képeznek
   (cos(wa)cos(wb), sin(wa)sin(wb)), amelyekből cos(w(a+b)) összerakható.
4. **Kiolvasás:** a neuronok egyváltozós hozzájárulásai kioltják egymást, a szorzat-tagok
   összeadódnak → a logitok a c = a+b helyen maximálisak.

## Korlátok
Egy feladat, egy architektúra (1 réteg, nincs LayerNorm), egy adatfelosztás, p = 113.
Az attention szerepe csak részben ismert. A küszöbök előre rögzített, de saját döntések.

## Nyitott kérdések
- Attention csere-szimmetria; mi viszi a maradék információt a szorzat-tagok nélkül?
- A közepes energiájú frekvenciák szerepe; késői loss-kiugrások (seed6, 38000. lépés).
- Megmaradnak-e ezek a mechanizmusok nagyobb modellekben és más feladatokon?

Részletek: JOURNAL.md, hypotheses/, experiments/.

---

# 2. fázis — kétrétegű modell (2026-10-09)

Egyetlen változó: n_layers 1 → 2 (424 064 paraméter).

| # | Állítás | Eredmény |
|---|---|---|
| H011 | A 2 rétegű modell is generalizál | SUPPORTED, 3/3 (5500–7300 lépés) |
| H012 | Embedding: Fourier-koncentráció (top-8 ≥ 0.70) | SUPPORTED, 3/3 (0.92–0.99) |
| H013 | A kulcsfrekvenciák szükségesek és elégségesek | SUPPORTED, 3/3 |

Az embedding-szintű mechanizmus átvihető. Eltérés (exploratív): a tanítás erősen
instabil, a koncentráció ingadozik. Nyitott: melyik réteg végzi a számítást.

## 2. fázis — munkamegosztás (EXP016 felfedezés, EXP018 megerősítés friss seedeken)

| # | Állítás | Eredmény |
|---|---|---|
| H014 | 1. réteg MLP-je szorzat-tagokat képez (≥ 0.10), a 0. rétegé a "=" pozíción nem (≤ 0.02) | SUPPORTED, 3/3 |
| H015 | A helyes válasz közvetlen logit-előnyének > 80%-a az 1. réteg MLP-jéből jön; a 0. rétegből ≈ 0 | SUPPORTED, 3/3 |
| H016 | 1. réteg: szorzat-tagok kellenek, egyváltozós tagok nem; 0. réteg: egyváltozós tagok kellenek | SUPPORTED, 3/3 |
| H017 | Mind a négy komponens (attn0, mlp0, attn1, mlp1) nélkülözhetetlen | SUPPORTED, 3/3 |

**Mechanizmus a 2 rétegű modellben:** a 0. réteg előkészít (számonként, szorzás nélkül),
az 1. réteg az 1 rétegű modellből ismert szorzat-mechanizmust futtatja. A mechanizmus
"magja" megmaradt, de egy előkészítő lépés került elé.
Nyitott: mit készít elő a 0. réteg; az erős tanítási instabilitás oka.
