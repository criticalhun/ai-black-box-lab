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
