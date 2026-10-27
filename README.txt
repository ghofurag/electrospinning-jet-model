KODE SIMULASI ELEKTROSPINNING v3 (baseline 16 kV)
=================================================
esjet.py            Model bead viskoelastik diskret (Reneker 2000), versi perbaikan program tim (CHANGELOG di akhir file).
calib16.py          Kalibrasi rho_q terhadap diameter SEM pada 16 kV:  python3 calib16.py 125 130 135
prod16.py           Semua variasi parameter (5 seed), konvergensi, run untuk gambar -> prod.json, rec.pkl
summarize16.py      Ringkasan dan eksponen sensitivitas
figs16.py           Gambar 1-5 naskah (sesuaikan path di bagian atas file)
conv.py             Uji konvergensi (versi 15 kV lama; versi 16 kV termasuk di prod16.py: grup convA/convC)
ukur_diameter_sem.py  Pengukuran otomatis diameter serat dari citra SEM (FWHM)
prod.json           Keluaran semua run (195 run)

Baseline: V=16 kV, h=12 cm, Q=0.4 mL/jam, mu=1.75 mPa.s, rho_q=130 C/m3 (I=14.4 nA), a0=3 um, L0=100 um.
Python 3.13, NumPy 2.5, Numba 0.68 (opsional), matplotlib, scipy, scikit-image.
