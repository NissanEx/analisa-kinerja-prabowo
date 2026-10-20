# Analisis Sentimen Publik terhadap Kinerja Pemerintahan Indonesia

Proyek ini menganalisis sentimen percakapan media sosial tentang program dan kinerja pemerintah Indonesia (bansos, BPJS/KIS, pendidikan, harga, infrastruktur, korupsi, layanan publik, dan ketenagakerjaan). Data dikumpulkan dari Facebook, Instagram, TikTok, dan Twitter/X, lalu diklasifikasikan menjadi positif, netral, dan negatif.

## Alur Analisis

1. **Scraping data**: Apify Client dengan daftar kata kunci program, kritik, dan meme/sindiran. Data dummy hanya dipakai sebagai cadangan dan ditandai di kolom `sumber`.
2. **Pembersihan**: buang teks pendek dan duplikat, rapikan metadata (author, followers, tanggal).
3. **Penandaan**: topik program (9 kategori, berbasis kata kunci) dan penanda meme/sindiran.
4. **Klasifikasi sentimen (ensemble)**: rata-rata probabilitas dari dua model:
   - `w11wo/indonesian-roberta-base-sentiment-classifier`
   - `cardiffnlp/twitter-xlm-roberta-base-sentiment`
5. **Koreksi sarkasme**: teks berlabel positif yang memuat pola sindiran (misalnya "katanya", "padahal", "klasik") diubah menjadi negatif.
6. **Ringkasan**: Indeks Sentimen Bersih (ISB), selang kepercayaan bootstrap, serta tabel dan grafik per platform, topik, dan meme vs non-meme.
7. **Laporan**: PDF otomatis dengan `fpdf2`.

## Rumus Utama

- Ensemble: `p(i) = (1/M) × Σ p_m(i)`, label = `argmax p(i)`
- Prediksi ragu-ragu bila skor maksimum < 0,55
- Indeks Sentimen Bersih: `ISB = rata-rata(s) × 50 + 50`, dengan positif = +1, netral = 0, negatif = −1 (50 = seimbang)
- Selang kepercayaan 95%: bootstrap persentil, 2.000 resampling

## Hasil Ringkas (167 data asli)

| Sentimen | Jumlah | Persentase |
|---|---|---|
| Positif | 26 | 15,6% |
| Netral | 79 | 47,3% |
| Negatif | 62 | 37,1% |

- **ISB = 39,2** (skala 0–100), menunjukkan kecenderungan negatif.
- Di antara teks yang beropini jelas, sekitar 70% negatif dan 30% positif.
- Instagram (47%) dan Twitter/X (40%) lebih negatif dibanding Facebook (11%), karena data Facebook berasal dari akun media berita dan didominasi netral.
- Seluruh konten meme/sindiran (14 data) bersentimen negatif.

## Keterbatasan

- Bukan survei representatif, dan sumber data bias ke platform dan kata kunci tertentu.
- Belum ada filter relevansi, sehingga sebagian berita non-pemerintahan ikut terhitung.
- Pelabelan topik berbasis kata kunci masih menimbulkan salah label.
- Deteksi sarkasme berbasis aturan dan belum divalidasi dengan data berlabel manusia (belum ada akurasi/F1).
- Hasil mengukur nada teks, bukan kepuasan publik yang sebenarnya.

## Cara Menjalankan

1. Buka `AnalisaKepuasanMasyarakat.ipynb` di Google Colab.
2. Isi `APIFY_TOKEN` dengan token Apify Anda sendiri. **Jangan unggah token ke GitHub.**
3. Jalankan sel secara berurutan hingga laporan PDF terbentuk.

## Teknologi

Python, Apify, pandas, Hugging Face Transformers, PyTorch, matplotlib, fpdf2.
