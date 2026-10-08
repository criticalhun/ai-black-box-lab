# Research Journal

## 2026-10-08 — EXP000 — Környezet felállítása (Fedora desktop)

### Cél
Izolált, reprodukálható Python/PyTorch környezet, GPU-támogatással.

### Hipotézis
Nincs (infrastruktúra-lépés).

### Megjegyzés
Tiszta újrakezdés. Egy korábbi, laptopos (CPU-only) próbálkozás
(2026-10-05) a gép terhelése miatt félbemaradt; abból semmilyen
eredményt nem használunk.

### Módszer
uv-val kezelt projekt; uv-managed Python 3.12 (független a rendszer
Python 3.14-étől); PyTorch a download.pytorch.org/whl/cu130 indexről.
A verziókat a `uv.lock` rögzíti.

### Környezet
- OS: Fedora Linux 44 Workstation, kernel 7.2.7-200.fc44.x86_64
- CPU: AMD Ryzen 9 5950X (16 mag / 32 szál; torch: 16 thread)
- RAM: 46 GiB
- GPU: NVIDIA GeForce RTX 4060, 8 GiB, compute capability 8.9
- Driver: 615.71.09 (max. támogatott CUDA: 13.4)
- Python: 3.12.14 (uv-managed)
- PyTorch: 2.14.1+cu130 | cuDNN: 9.24.0 | NumPy: 2.5.3
- uv: 0.12.19 (Fedora csomag)
- Részletek: `notes/env_check_*.txt`

### Parancsok
    sudo dnf install -y uv gh
    uv python install 3.12
    uv python pin 3.12
    uv add torch numpy pyyaml
    # reprodukálás máshol:
    uv python install 3.12 && uv sync

### Eredmények
- `torch.cuda.is_available()` = True, 4096×4096 GPU matmul lefut
- Azonos seed → azonos `torch.randn` kimenet CPU-n és GPU-n is (True)

### Felmerült akadályok (reprodukálhatóság miatt rögzítve)
- PackageKit "command not found" kezelője interaktív telepítést indított
  egy bemásolt blokk közepén → megszakítva, uv/gh dnf-fel telepítve.
- ble.sh: bemásolt több soros blokk multiline módba kerül → futtatás: Ctrl+J.
- A Fedora uv csomagja `python-downloads = manual` → `uv python install 3.12`.

### Értelmezés
A környezet alkalmas kis modellek GPU-s és CPU-s tanítására.

### Bizonytalanság
- A seed-teszt csak az RNG determinizmusát mutatja, nem egy teljes tanítás
  bit-pontos reprodukálhatóságát → EXP001-ben külön ellenőrizzük.
- GPU-n egyes műveletek alapból nem determinisztikusak; a szükséges
  beállítások (deterministic algorithms, CUBLAS workspace, TF32 tiltás)
  hatását még nem igazoltuk.

### Következő lépés
H001 és EXP001 preregisztrálása, majd a modell implementálása.

## 2026-10-08 — EXP001 → H001

### Cél
Létrehozni egy kis Transformert, amely a `(a+b) mod 113` feladatot
generalizálja (nem csak memorizálja) — a későbbi belső vizsgálatok tárgyát.

### Hipotézis
H001: a 30%-on tanított 1 rétegű modell a maradék 70%-on ≥ 99% pontosságot ér el
(kritérium: mindhárom seed).

### Módszer
Preregisztráció: commit a188156. Kód: commit f995dc3.
Full-batch AdamW (lr 1e-3, wd 1.0), 40 000 lépés, seed 0/1/2,
fix adatfelosztás (data_seed 598, split sha256 8de4382ea6c7…),
+ seed 0 újrafuttatás determinizmus-ellenőrzésre.

### Környezet
Mint EXP000 (fedora, RTX 4060, torch 2.14.1+cu130), CUDA, deterministic algorithms,
CUBLAS_WORKSPACE_CONFIG=:4096:8, float32 matmul precision "highest" (TF32 tiltva).

### Parancsok
    uv run python -m tests.test_model
    uv run python -m tests.test_data
    uv run python experiments/EXP001/train.py --config experiments/EXP001/config.yaml --seed {0,1,2}
    uv run python experiments/EXP001/train.py --config experiments/EXP001/config.yaml --seed 0 --tag rerun
    cmp experiments/EXP001/results/seed0/metrics.csv experiments/EXP001/results/seed0_rerun/metrics.csv

### Eredmények
| seed | train acc ≥ 0.99 | test acc ≥ 0.99 | végső test acc |
|---|---|---|---|
| 0 | 200 | 10400 | 1.0000 |
| 1 | 200 | 8500 | 1.0000 |
| 2 | 200 | 10900 | 1.0000 |

- 0. lépés loss: 4.761 / 4.774 / 4.767 (előrejelzés: ≈ ln 114 = 4.736) ✓
- A memorizációs fázisban a test loss ~24–25-re nőtt (≈ 5× a véletlen szint).
- A test loss már ~2–3 ezer lépéstől csökkent, miközben a test acc lassan nőtt;
  az acc-ugrás kb. 8–11 ezer lépésnél történt.
- seed0 és seed0_rerun: metrics.csv bit-azonos, végső súly-hash azonos (8085b4b645d6).

### Értelmezés
- **H001: SUPPORTED** (3/3 seed, csak erre a konfigurációra).
- A lefutás a "grokking" mintát követi (gyors memorizáció, késleltetett generalizáció).
- A memorizáló modell a nem látott párokon magabiztosan téved (test loss ≫ véletlen).
- A test loss fokozatos csökkenése az acc-ugrás előtt összhangban van azzal az
  irodalmi állítással (Nanda et al. 2023), hogy a generalizáló megoldás fokozatosan
  épül fel — ezt a belső struktúra mérésével NEM ellenőriztük.

### Bizonytalanság
- 3 seed, 1 felosztás, 1 konfiguráció; 100 lépéses felbontás.
- Késői fázis: seed0 test loss lassan nő (5.3e-7 → 7.2e-7), seed2 a végén kissé
  megugrik. ~1e-7 szinten a float32 logit-pontosság számít — nem tudjuk, érdemi-e.
- A determinizmus csak azonos gépen / szoftverkörnyezetben igazolt.

### Következő lépés
A betanított modell belsejének vizsgálata: H002 preregisztrálása
(az embeddingek szerkezete), majd kauzális teszt.

## 2026-10-08 — EXP002 → H002

### Cél
Megnézni, hogy a betanított modellek számembeddingjei Fourier-bázisban koncentráltak-e.

### Hipotézis
H002: végső W_E[0:113] energiájának ≥ 70%-a a top-8 nem-konstans frekvencián (3/3 seed),
kontroll (init) < 0.30.

### Módszer
Preregisztráció: commit abbd0b9 (a spektrumok kiszámolása ELŐTT).
Mérőeszköz: src/fourier.py, kalibrálva (tests/test_fourier.py: véletlen ≈ 0.162,
ismert {5,17} szerkezet visszanyerve). Elemzés: commit 98843a4.
Integritás: minden checkpoint hash-e egyezik az EXP001 summary.json-nal.

### Parancsok
    uv run python -m tests.test_fourier
    uv run python experiments/EXP002/analyze.py --config experiments/EXP002/config.yaml

### Eredmények
| seed | kontroll | végső top-8 | top-8 frekvenciák |
|---|---|---|---|
| 0 | 0.160 | 0.994 | 9, 18, 19, 28, 30, 33, 38, 47 |
| 1 | 0.160 | 0.995 | 10, 20, 23, 27, 31, 41, 51, 54 |
| 2 | 0.162 | 0.996 | 1, 2, 4, 9, 17, 18, 27, 56 |

Konstans komponens aránya: ~0.000 mindhárom seednél.
Exploratív (results/trajectory.csv): a top-8 arány már 2000 lépésnél ~0.24–0.26,
6000-nél 0.32–0.41, a test acc ugrása előtt; az elsőként generalizáló seed (1)
koncentrációja nőtt leggyorsabban, az utolsóé (2) leglassabban; a koncentráció a
100% test acc után is nőtt (seed0: 0.926 @12k → 0.991 @18k).

### Értelmezés
- **H002: SUPPORTED.** Az embeddingek erősen Fourier-ritkák.
- A seedek különböző frekvenciákat választanak — összhangban azzal, hogy Z_113-ban
  nincs kitüntetett nem-nulla frekvencia (magyarázat, nem mérés).
- Exploratív: a Fourier-struktúra fokozatosan épül és megelőzi az acc-ugrást;
  az ugrás után is "tisztul". Összhangban Nanda et al. 2023-mal — 3 seed korrelációja,
  nem ok-okozat.

### Bizonytalanság
- Nem tudjuk, hány frekvencia "igazán" fontos a 8-ból (szándékosan nem néztük a
  H003 rögzítése előtt).
- A koncentráció nem bizonyítja, hogy a modell használja ezeket a frekvenciákat.

### Következő lépés
H003: kauzális teszt — kulcsfrekvenciák eltávolítása / megtartása W_E-ben, kontrollal.
