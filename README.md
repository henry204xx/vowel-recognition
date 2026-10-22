# Pengenalan Vowel Bahasa Indonesia dengan DTW dan Model Gaussian

Proyek praktikum untuk mengenali `/a/`, `/i/`, `/u/`, `/e/`, dan `/o/` menggunakan template DTW dan Gaussian multivariat berbasis MFCC.

## Instalasi

Gunakan Python 3.10+.

```bash
pip install -r requirements.txt
pip install -e .
```

## Struktur data

Simpan rekaman langsung di `data/S01/a/train.wav` dan `data/S01/a/test.wav`, diulang untuk semua vokal dan pembicara. Konfigurasi pembicara berada di [config.yaml](config.yaml). Cache fitur (bila diaktifkan) disimpan di `results/feature_cache`, terpisah dari data rekaman.

## Menjalankan lewat notebook

Buka [vowel_recognition_pipeline.ipynb](vowel_recognition_pipeline.ipynb) di VS Code atau Jupyter, lalu jalankan sel dari atas ke bawah. Notebook memvalidasi data, memperlihatkan contoh prapemrosesan/MFCC, menjalankan kedua sistem, menampilkan hasil, dan menyimpannya ke `results/`; skrip peluncur terpisah tidak diperlukan.

## Pengujian kode

```bash
pytest
```

Hasil CSV/JSON berada di `results/tables` dan `results`, sedangkan grafik berada di `results/figures`. Evaluasi *open speaker* hanya tersedia bila pembicara mandiri dikonfigurasi dan rekamannya ada; nilai tidak akan dibuat-buat.
