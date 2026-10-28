# Tugas Pemrograman Kelompok II - Natural Language Processing (TI6073)

Repositori ini berisi implementasi sistem **Mesin Pencari Dokumen Ilmiah berbasis TF-IDF dan Cosine Similarity** dengan korpus bertopik **Automated Short Answer Grading (ASAG)** berbahasa Inggris.

* **Program Studi:** Informatika, Fakultas Teknologi Informasi  
* **Institusi:** Universitas Kristen Duta Wacana (UKDW), Yogyakarta  

---

## 1. Identitas Anggota Kelompok

| No | NIM | Nama Lengkap | Alokasi Dokumen PDF |
|:---:|:---:|:---|:---|
| 1 | **71231050** | **Samuel Natanael** | [`PDF/Sam/`](PDF/Sam/) (5 Dokumen) |
| 2 | **71241154** | **Indira Mai Narwastu** | [`PDF/Indira/`](PDF/Indira/) (5 Dokumen) |
| 3 | **71231021** | **Christian Nicholas Saputra** | [`PDF/NicSap/`](PDF/NicSap/) (5 Dokumen) |

---

## 2. Struktur Direktori Repositori

```text
├── PDF/                                      # Berkas artikel ilmiah asli (.pdf)
│   ├── Indira/                               # 5 PDF artikel anggota Indira
│   ├── Sam/                                  # 5 PDF artikel anggota Samuel
│   └── NicSap/                               # 5 PDF artikel anggota NicSap
├── corpus/                                   # Teks (.txt) hasil ekstraksi 15 PDF
├── asag_search_engine.py                     # Program utama Python (CLI & Streamlit Web UI)
├── README.txt                                # Dokumentasi resmi untuk pengumpulan E-Class (ZIP)
└── README.md                                 # Dokumentasi Markdown untuk repositori GitHub
```

---

## 3. Daftar Dokumen Korpus & Sumber (15 Dokumen ASAG)

### A. Dokumen Indira Mai Narwastu (`PDF/Indira/`)
1. **A Hybrid Approach for Automated Short Answer Grading**  
   * Mustafa Kaya, Ilyas Cicekli (2024) — *IEEE Access*, Vol. 12, pp. 96332–96341.  
   * DOI: [10.1109/ACCESS.2024.3420890](https://doi.org/10.1109/ACCESS.2024.3420890)
2. **Automated Short-Answer Grading using Semantic Similarity based on Word Embedding**  
   * Fetty Fitriyanti Lubis et al. (2021) — *International Journal of Technology (IJTech)*, Vol. 12(3), pp. 571–581.  
   * DOI: [10.14716/ijtech.v12i3.4651](https://doi.org/10.14716/ijtech.v12i3.4651)
3. **Comparative Evaluation of Transformer-Based Models for Automated Short-Answer Grading**  
   * Temidayo O. Omotehinwa et al. (2025) — *NIPES Journal of Science and Technology Research*, Vol. 7, pp. 843–849.  
   * DOI: [10.37933/nipes/7.4.2025.SI97](https://doi.org/10.37933/nipes/7.4.2025.SI97)
4. **Performance Analysis of Transformer Based Models for Automatic Short Answer Grading**  
   * Mrs. Rupal Chaudhari, Dr. Manish Patel (2025) — *Journal of Information Systems Engineering and Management (JISEM)*, Vol. 10(51s), pp. 321–334.  
   * URL: [JISEM Journal](https://www.jisem-journal.com/)
5. **Performance of the Pre-Trained Large Language Model GPT-4 on Automated Short Answer Grading**  
   * Gerd Kortemeyer (2024) — *Discover Artificial Intelligence*, Vol. 4, Article 47.  
   * DOI: [10.1007/s44163-024-00147-y](https://doi.org/10.1007/s44163-024-00147-y)

### B. Dokumen Samuel Natanael (`PDF/Sam/`)
6. **Automatic Short Answer Grading With SemSpace Sense Vectors and MaLSTM**  
   * Cagatay Neftali Tulu, Ozge Ozkaya, Umut Orhan (2021) — *IEEE Access*, Vol. 9, pp. 19270–19280.  
   * DOI: [10.1109/ACCESS.2021.3054346](https://doi.org/10.1109/ACCESS.2021.3054346)
7. **Joint Multi-Domain Learning for Automatic Short Answer Grading**  
   * Swarnadeep Saha et al. (2019) — *arXiv:1902.09183*.  
   * URL: [https://arxiv.org/abs/1902.09183](https://arxiv.org/abs/1902.09183)
8. **Language Models are Few-Shot Graders**  
   * Chenyan Zhao, Mariana Silva, Seth Poulsen (2025) — *arXiv:2502.13337*.  
   * URL: [https://arxiv.org/abs/2502.13337](https://arxiv.org/abs/2502.13337)
9. **Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments**  
   * Michael Mohler, Razvan Bunescu, Rada Mihalcea (2011) — *ACL HLT 2011*, pp. 752–762.  
   * URL: [ACL Anthology P11-1076](https://aclanthology.org/P11-1076/)
10. **RATAS A Generative AI Framework for Explainable and Scalable Automated Answer Grading**  
    * Masoud Safilian, Amin Beheshti, Stephen Elbourn (2025) — *arXiv:2505.23818*.  
    * URL: [https://arxiv.org/abs/2505.23818](https://arxiv.org/abs/2505.23818)

### C. Dokumen Christian Nicholas Saputra (`PDF/NicSap/`)
11. **Automatic Short Answer Grading via Multiway Attention Networks**  
    * Tiaoqiao Liu et al. (2019) — *arXiv:1909.10166* / *NLPTEA 2019*.  
    * URL: [https://arxiv.org/abs/1909.10166](https://arxiv.org/abs/1909.10166)
12. **Automatic Short Math Answer Grading via In-context Meta-learning**  
    * Mengxue Zhang et al. (2022) — *arXiv:2205.15219*.  
    * URL: [https://arxiv.org/abs/2205.15219](https://arxiv.org/abs/2205.15219)
13. **Fast and Easy Short Answer Grading with High Accuracy**  
    * Md Arafat Sultan, Cristobal Salazar, Tamara Sumner (2016) — *NAACL-HLT 2016*, pp. 1070–1075.  
    * URL: [ACL Anthology N16-1123](https://aclanthology.org/N16-1123/)
14. **Grade Guard A Smart System for Short Answer Automated Grading**  
    * Niharika Dadu, Harsh Vardhan Singh, Romi Banerjee (2025) — *arXiv:2504.01253*.  
    * URL: [https://arxiv.org/abs/2504.01253](https://arxiv.org/abs/2504.01253)
15. **Survey on Automated Short Answer Grading with Deep Learning from Word Embeddings to Transformers**  
    * Stefan Haller et al. (2022) — *arXiv:2204.03503*.  
    * URL: [https://arxiv.org/abs/2204.03503](https://arxiv.org/abs/2204.03503)

---

## 4. Instalasi & Cara Menjalankan

### Persyaratan Pustaka
```bash
pip install nltk scikit-learn pandas numpy pymupdf streamlit
```

### Eksekusi Program
1. **Mode Terminal / CLI:**
   ```bash
   python asag_search_engine.py
   ```

2. **Mode Antarmuka Web (Streamlit):**
   ```bash
   streamlit run asag_search_engine.py
   ```
