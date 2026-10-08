# EXP001 — 1 rétegű Transformer, moduláris összeadás

**Hipotézis:** H001
**Státusz:** PREREGISTERED (még nem futott)

## Cél
Létrehozni a vizsgálat tárgyát: egy kis Transformert, amely a feladatot
generalizálja, nem csak memorizálja. Ez a kísérlet NEM értelmez semmit
a belső működésről.

## Miért moduláris összeadás?
Ismert, publikált mechanizmus van rá (Nanda et al., 2023, "Progress measures
for grokking via mechanistic interpretability"). Ez kalibrációs pont: ha a
későbbi elemző eszközeink nem találják meg az ismert mechanizmust,
az eszközeinkben van a hiba. A sima (nem moduláris) összeadás: EXP002.

## Feladat
Bemenet: `[a, b, =]`, cél: `(a + b) mod 113`. Minden szám egyetlen token.
Teljes tér: 113² = 12 769 pár → 30% train (fix felosztás), 70% test.

## Futtatás
Minden EXP001-futás ugyanazon a gépen (fedora, RTX 4060), CUDA-n,
determinisztikus módban. CPU és GPU eredményei bitre nem összevethetők.

## Mérések
- train/test loss és accuracy `log_every` lépésenként → `results/seed{N}/metrics.csv`
- checkpointok → `results/seed{N}/checkpoints/` (nincs gitben)
- determinizmus-ellenőrzés: seed 0 kétszer

## Amit az eredmény NEM bizonyít
A magas teszt-pontosság csak azt mutatja, hogy van mit vizsgálni.
A mechanizmusra vonatkozó állítások későbbi hipotézisek (H002+) tárgyai.
