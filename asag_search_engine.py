"""
================================================================================
TUGAS PEMROGRAMAN KELOMPOK II - NATURAL LANGUAGE PROCESSING (TI6073)
Topik Korpus : Automated Short Answer Grading (ASAG)
Fitur Utama  :
1. Pembacaan Korpus Dokumen (.txt & ekstraksi otomatis dari .pdf)
2. Praproses Teks: Normalisasi (Case Folding & Hapus Tanda Baca/Angka),
   Tokenisasi (NLTK word_tokenize), Stopword Removal (NLTK stopwords),
   dan Stemming (NLTK PorterStemmer / Sastrawi).
3. Perhitungan Frekuensi Token per Dokumen (Term Frequency / TF dengan FreqDist)
   serta Document Frequency (DF).
4. Pembobotan TF-IDF (sklearn TfidfVectorizer), Penyimpanan Index, dan
   Transformasi Transpose (.transpose()) agar Term menjadi Baris & Dokumen
   menjadi Kolom.
5. Pencarian Kueri Frasa Pengguna: Praproses Kueri, Pembobotan TF-IDF Kueri,
   Perhitungan Kemiripan Cosine (Cosine Similarity), dan Pengurutan Dokumen
   dari Nilai Kemiripan Tertinggi.
6. BONUS 2.5%: Antarmuka Mesin Pencari Sederhana berbasis Web (Streamlit)
   sekaligus mendukung Mode Terminal / CLI Interaktif.
================================================================================
"""

import os
import re
import sys
import string
import subprocess
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# INISIALISASI SUMBER DAYA NLTK
# ==============================================================================
def ensure_nltk_resources() -> None:
    """Memastikan paket data NLTK ('punkt', 'punkt_tab', 'stopwords') tersedia."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path_check, pkg_name in resources:
        try:
            nltk.data.find(path_check)
        except LookupError:
            nltk.download(pkg_name, quiet=True)


ensure_nltk_resources()


# ==============================================================================
# KONFIGURASI DIREKTORI & PEMBACAAN KORPUS (.PDF & .TXT)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")

# Daftar nama berkas non-korpus yang harus diabaikan saat memindai folder utama
IGNORED_FILES = {
    "06 practice session.pdf",
    "readme.txt",
    "requirements.txt",
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Mengekstrak teks dari berkas PDF menggunakan PyMuPDF (fitz) atau PyPDF2."""
    text_pages = []
    # Coba gunakan PyMuPDF (fitz) terlebih dahulu karena lebih cepat & akurat
    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_pages.append(page.get_text("text"))
        return "\n".join(text_pages)
    except ImportError:
        pass

    # Fallback ke PyPDF2
    try:
        import PyPDF2

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        return "\n".join(text_pages)
    except Exception as exc:
        print(f"[Peringatan] Gagal mengekstrak PDF '{pdf_path}': {exc}")
        return ""


def sync_and_load_corpus(corpus_dir: str = CORPUS_DIR) -> Dict[str, Dict[str, Any]]:
    """
    Memindai seluruh folder dan subfolder anggota kelompok (seperti Indira/, Sam/,
    NicSap/) menggunakan os.walk() sesuai modul kuliah.
    Jika ada berkas .pdf artikel yang belum memiliki versi .txt di `corpus/`,
    otomatis diekstrak ke `corpus/*.txt`.
    Mengembalikan dictionary dokumen dengan kunci Kode Dokumen ('D01', 'D02', ...).
    """
    os.makedirs(corpus_dir, exist_ok=True)

    # 1. Pindai seluruh folder kerja & subfolder menggunakan os.walk()
    pdf_source_map: Dict[str, str] = {}
    for path, subdirs, files in os.walk(BASE_DIR):
        # Jangan memindai folder sistem/cache
        subdirs[:] = [
            d
            for d in subdirs
            if not d.startswith(".") and d not in {"__pycache__", "venv", ".venv"}
        ]
        for fname in sorted(files):
            lower_name = fname.lower()
            if lower_name in IGNORED_FILES:
                continue
            full_path = os.path.join(path, fname)
            if lower_name.endswith(".pdf"):
                base_name = os.path.splitext(fname)[0]
                rel_folder = os.path.relpath(path, BASE_DIR)
                pdf_source_map[base_name] = (
                    rel_folder if rel_folder != "." else "Root"
                )
                txt_target = os.path.join(corpus_dir, f"{base_name}.txt")
                if not os.path.exists(txt_target) or os.path.getsize(txt_target) == 0:
                    extracted = extract_text_from_pdf(full_path)
                    if extracted.strip():
                        with open(txt_target, "w", encoding="utf-8") as out_f:
                            out_f.write(extracted)

    # Hapus berkas .txt yatim (bekas nama PDF lama yang sudah di-rename/dihapus)
    if pdf_source_map:
        for fname in os.listdir(corpus_dir):
            if fname.lower().endswith(".txt"):
                bname = os.path.splitext(fname)[0]
                if bname not in pdf_source_map:
                    try:
                        os.remove(os.path.join(corpus_dir, fname))
                    except OSError:
                        pass

    # 2. Baca seluruh berkas .txt di dalam CORPUS_DIR
    txt_files = sorted(
        [
            f
            for f in os.listdir(corpus_dir)
            if f.lower().endswith(".txt") and f.lower() not in IGNORED_FILES
        ]
    )

    documents: Dict[str, Dict[str, Any]] = {}
    for idx, fname in enumerate(txt_files, start=1):
        doc_code = f"D{idx:02d}"
        fpath = os.path.join(corpus_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
            raw_content = fh.read()

        base_name = os.path.splitext(fname)[0]
        clean_title = base_name.replace("_", " ").strip()
        owner_folder = pdf_source_map.get(base_name, "corpus")
        documents[doc_code] = {
            "code": doc_code,
            "filename": fname,
            "owner": owner_folder,
            "filepath": fpath,
            "title": clean_title,
            "raw_text": raw_content,
        }

    return documents


# ==============================================================================
# PRAPROSES TEKS (NORMALISASI, TOKENISASI, STOPWORD REMOVAL, STEMMING)
# ==============================================================================
class TextPreprocessor:
    """
    Kelas untuk melakukan tahapan praproses teks sesuai instruksi tugas:
    1. Normalisasi teks (case folding + penghilangan tanda baca & angka)
    2. Tokenisasi (NLTK word_tokenize)
    3. Penghilangan Stopword (NLTK stopwords)
    4. Stemming (PorterStemmer untuk Bahasa Inggris / Sastrawi untuk Indonesia)
    """

    def __init__(self, language: str = "english"):
        self.language = language.lower()
        self.stop_words = set(stopwords.words(self.language))

        if self.language == "indonesian":
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

                factory = StemmerFactory()
                self.idn_stemmer = factory.create_stemmer()
                self.en_stemmer = None
            except ImportError:
                self.idn_stemmer = None
                self.en_stemmer = PorterStemmer()
        else:
            self.en_stemmer = PorterStemmer()
            self.idn_stemmer = None

    def normalize(self, text: str) -> str:
        """
        Normalisasi teks:
        - Mengubah huruf besar menjadi huruf kecil (case folding)
        - Menghapus URL, angka, dan tanda baca (string.punctuation & RegEx)
        """
        # Case folding
        text = text.lower()
        # Hapus URL jika ada
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Hapus karakter non-alfabet dan angka sesuai modul: re.sub(r'[\W\s(0-9)]+', ' ', ...)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"[\W\s0-9_]+", " ", text)
        return text.strip()

    def tokenize(self, normalized_text: str) -> List[str]:
        """Melakukan tokenisasi menggunakan nltk.word_tokenize()."""
        return word_tokenize(normalized_text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Menghilangkan stopword dan token satu huruf atau karakter non-ASCII."""
        return [
            w
            for w in tokens
            if w not in self.stop_words and len(w) > 1 and w.isalpha() and w.isascii()
        ]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Melakukan proses stemming pada daftar token."""
        if self.language == "indonesian" and self.idn_stemmer is not None:
            return [self.idn_stemmer.stem(w) for w in tokens]
        return [self.en_stemmer.stem(w) for w in tokens]

    def preprocess(self, text: str) -> Dict[str, Any]:
        """Menjalankan seluruh tahapan praproses teks dan mengembalikan hasilnya."""
        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)
        filtered_tokens = self.remove_stopwords(tokens)
        stemmed_tokens = self.stem_tokens(filtered_tokens)
        clean_string = " ".join(stemmed_tokens)
        return {
            "normalized": normalized,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "stemmed_tokens": stemmed_tokens,
            "clean_text": clean_string,
        }


# ==============================================================================
# MESIN INDEKS TF-IDF & PENCARIAN COSINE SIMILARITY
# ==============================================================================
class ASAGSearchEngine:
    """
    Mesin Pencari Korpus Artikel Ilmiah ASAG berbasis TF-IDF dan Cosine Similarity.
    Menyimpan:
    - `doks_tf`            : Dictionary of FreqDist (frekuensi token tiap dokumen)
    - `doc_freq`           : Dictionary Document Frequency (DF) tiap term
    - `df_tfidf_docs`      : DataFrame TF-IDF (Baris = Dokumen, Kolom = Term)
    - `df_tfidf_transpose` : DataFrame Index Hasil Transpose (Baris = Term, Kolom = Dokumen)
    """

    def __init__(self, corpus_dir: str = CORPUS_DIR, language: str = "english"):
        self.corpus_dir = corpus_dir
        self.language = language
        self.preprocessor = TextPreprocessor(language=language)

        self.documents: Dict[str, Dict[str, Any]] = {}
        self.doc_codes: List[str] = []
        self.doks_tf: Dict[str, nltk.FreqDist] = {}
        self.doc_freq: Dict[str, int] = {}

        self.tfidf_vec: TfidfVectorizer = TfidfVectorizer()
        self.tfidf_wm = None
        self.tfidf_wm_transposed = None
        self.tfidf_terms: np.ndarray = np.array([])

        # Index dalam bentuk DataFrame dan Dictionary
        self.df_tfidf_docs: pd.DataFrame = pd.DataFrame()
        self.df_tfidf_transpose: pd.DataFrame = pd.DataFrame()
        self.inverted_index: Dict[str, Dict[str, float]] = {}

    def build_index(self) -> None:
        """
        Membaca korpus, melakukan praproses, menghitung TF & DF, menghitung
        bobot TF-IDF, menyimpan ke dalam index, dan melakukan transpose.
        """
        self.documents = sync_and_load_corpus(self.corpus_dir)
        self.doc_codes = list(self.documents.keys())

        if not self.doc_codes:
            raise RuntimeError(
                f"Tidak ada dokumen ditemukan di '{self.corpus_dir}'."
            )

        clean_corpus: List[str] = []
        self.doks_tf = {}
        df_counter: Counter = Counter()

        # 1. Praproses setiap dokumen & hitung TF (FreqDist) serta DF
        for code in self.doc_codes:
            raw_text = self.documents[code]["raw_text"]
            prep = self.preprocessor.preprocess(raw_text)
            self.documents[code].update(prep)

            stemmed_tokens = prep["stemmed_tokens"]
            # Hitung frekuensi tiap token di tiap dokumen menggunakan nltk.FreqDist
            term_freq = nltk.FreqDist(stemmed_tokens)
            self.doks_tf[code] = term_freq

            # Hitung Document Frequency (DF): tiap term unik dalam dokumen dihitung 1
            for unique_term in set(stemmed_tokens):
                df_counter[unique_term] += 1

            clean_corpus.append(prep["clean_text"])

        # Simpan Document Frequency (DF) terurut berdasarkan abjad term
        self.doc_freq = dict(sorted(df_counter.items(), key=lambda x: x[0]))

        # 2. Pembobotan TF-IDF menggunakan sklearn TfidfVectorizer
        self.tfidf_vec = TfidfVectorizer()
        self.tfidf_wm = self.tfidf_vec.fit_transform(clean_corpus)
        self.tfidf_terms = self.tfidf_vec.get_feature_names_out()

        # DataFrame TF-IDF standar (Baris = Kode Dokumen, Kolom = Term/Token)
        self.df_tfidf_docs = pd.DataFrame(
            data=self.tfidf_wm.toarray(),
            index=self.doc_codes,
            columns=self.tfidf_terms,
        )

        # 3. Gunakan fungsi transpose() pada matriks sklearn agar:
        #    Nama/Kode Dokumen menjadi KOLOM dan Term/Token menjadi BARIS
        self.tfidf_wm_transposed = self.tfidf_wm.transpose()
        self.df_tfidf_transpose = pd.DataFrame(
            data=self.tfidf_wm_transposed.toarray(),
            index=self.tfidf_terms,
            columns=self.doc_codes,
        )
        self.df_tfidf_transpose.index.name = "Term / Token"

        # Simpan juga dalam struktur data dictionary index (term -> {doc_code: tfidf})
        self.inverted_index = {}
        transposed_array = self.tfidf_wm_transposed.toarray()
        for t_idx, term in enumerate(self.tfidf_terms):
            row_vals = transposed_array[t_idx]
            self.inverted_index[term] = {
                self.doc_codes[d_idx]: float(row_vals[d_idx])
                for d_idx in range(len(self.doc_codes))
                if row_vals[d_idx] > 0.0
            }

    def get_tf_df_summary_table(self) -> pd.DataFrame:
        """
        Membuat tabel ringkasan frekuensi tiap token di tiap dokumen (TF)
        beserta Document Frequency (DF) untuk seluruh term di dalam index.
        """
        tf_matrix = np.zeros((len(self.tfidf_terms), len(self.doc_codes)), dtype=int)
        for d_idx, code in enumerate(self.doc_codes):
            fdist = self.doks_tf[code]
            for t_idx, term in enumerate(self.tfidf_terms):
                tf_matrix[t_idx, d_idx] = fdist.get(term, 0)

        df_table = pd.DataFrame(
            data=tf_matrix,
            index=self.tfidf_terms,
            columns=[f"TF_{c}" for c in self.doc_codes],
        )
        df_table["Total_TF"] = tf_matrix.sum(axis=1)
        df_table["DF (Doc Freq)"] = [
            self.doc_freq.get(term, 0) for term in self.tfidf_terms
        ]
        df_table.index.name = "Term / Token"
        return df_table

    def search(self, query_phrase: str) -> Dict[str, Any]:
        """
        Memproses kueri frasa dari pengguna:
        1. Praproses kueri (normalisasi, tokenisasi, stopword removal, stemming)
        2. Hitung bobot TF-IDF kueri menggunakan vectorizer yang telah dilatih
        3. Hitung kemiripan kueri dengan tiap dokumen menggunakan Cosine Similarity
        4. Sortir dokumen berdasarkan nilai kemiripan tertinggi
        """
        prep_query = self.preprocessor.preprocess(query_phrase)
        clean_query = prep_query["clean_text"]

        # Hitung bobot TF-IDF kueri
        query_tfidf_wm = self.tfidf_vec.transform([clean_query])
        query_tfidf_array = query_tfidf_wm.toarray()[0]

        # Ambil bobot TF-IDF untuk tiap term yang muncul di kueri
        query_term_weights: Dict[str, float] = {}
        for token in set(prep_query["stemmed_tokens"]):
            if token in self.tfidf_vec.vocabulary_:
                idx = self.tfidf_vec.vocabulary_[token]
                query_term_weights[token] = float(query_tfidf_array[idx])
            else:
                query_term_weights[token] = 0.0

        # Hitung Cosine Similarity antara kueri dan seluruh dokumen di korpus
        cos_sim_scores = cosine_similarity(query_tfidf_wm, self.tfidf_wm)[0]

        # Buat daftar hasil dan sortir berdasarkan nilai kemiripan tertinggi (descending)
        ranked_results = []
        for idx, code in enumerate(self.doc_codes):
            doc_info = self.documents[code]
            score = float(cos_sim_scores[idx])

            # Detail kontribusi term kueri pada dokumen ini
            matched_terms = {}
            for token in set(prep_query["stemmed_tokens"]):
                if token in self.df_tfidf_transpose.index:
                    w_doc = float(self.df_tfidf_transpose.loc[token, code])
                    if w_doc > 0:
                        matched_terms[token] = {
                            "tf_in_doc": int(self.doks_tf[code].get(token, 0)),
                            "tfidf_in_doc": w_doc,
                            "tfidf_in_query": query_term_weights.get(token, 0.0),
                        }

            ranked_results.append(
                {
                    "rank": 0,
                    "code": code,
                    "filename": doc_info["filename"],
                    "title": doc_info["title"],
                    "similarity": score,
                    "matched_terms": matched_terms,
                    "snippet": self._build_snippet(
                        doc_info["raw_text"], prep_query["filtered_tokens"]
                    ),
                }
            )

        ranked_results.sort(key=lambda item: item["similarity"], reverse=True)
        for rank_idx, item in enumerate(ranked_results, start=1):
            item["rank"] = rank_idx

        return {
            "raw_query": query_phrase,
            "preprocessed_query": prep_query,
            "query_tfidf_weights": query_term_weights,
            "results": ranked_results,
        }

    @staticmethod
    def _build_snippet(raw_text: str, query_tokens: List[str], max_len: int = 280) -> str:
        """Membuat cuplikan teks dokumen yang relevan dengan kueri."""
        clean_lines = " ".join(raw_text.split())
        if not clean_lines:
            return ""
        lower_text = clean_lines.lower()
        for tok in query_tokens:
            pos = lower_text.find(tok.lower())
            if pos != -1:
                start = max(0, pos - 80)
                end = min(len(clean_lines), pos + max_len - 80)
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(clean_lines) else ""
                return f"{prefix}{clean_lines[start:end]}{suffix}"
        return clean_lines[:max_len] + ("..." if len(clean_lines) > max_len else "")


# ==============================================================================
# MODE TERMINAL / CLI INTERAKTIF
# ==============================================================================
def run_cli_mode() -> None:
    """Menjalankan alur lengkap program di Terminal sesuai instruksi tugas."""
    sep = "=" * 80
    print(sep)
    print("   SISTEM TEMU KEMBALI INFORMASI (SEARCH ENGINE) KORPUS ARTIKEL ILMIAH ASAG")
    print("             Metode: Pembobotan TF-IDF & Kemiripan Cosine")
    print(sep)

    engine = ASAGSearchEngine(corpus_dir=CORPUS_DIR, language="english")
    engine.build_index()

    # 1. Tampilkan daftar dokumen dalam korpus
    print(f"\n[1] DAFTAR DOKUMEN KORPUS ({len(engine.doc_codes)} Dokumen Ditemukan):")
    print("-" * 80)
    for code in engine.doc_codes:
        doc = engine.documents[code]
        total_raw_tok = len(doc["tokens"])
        total_stem_tok = len(doc["stemmed_tokens"])
        unique_terms = len(engine.doks_tf[code])
        print(
            f"  {code} | {doc['filename']}\n"
            f"      -> Token Awal: {total_raw_tok:,} | Setelah Stopword & Stemming: "
            f"{total_stem_tok:,} | Term Unik: {unique_terms:,}"
        )

    # 2. Tampilkan sampel hasil praproses & frekuensi token (TF) per dokumen
    print(f"\n[2] FREKUENSI TOKEN TIAP DOKUMEN (TF) & DOCUMENT FREQUENCY (DF):")
    print("-" * 80)
    for code in engine.doc_codes:
        top_5 = engine.doks_tf[code].most_common(5)
        top_str = ", ".join([f"{k}:{v}" for k, v in top_5])
        print(f"  {code} (Top 5 Term Frequency) -> {top_str}")

    tf_df_table = engine.get_tf_df_summary_table()
    top_df_terms = tf_df_table.sort_values(
        by=["DF (Doc Freq)", "Total_TF"], ascending=False
    ).head(15)
    print("\n  15 Term Teratas Berdasarkan Document Frequency (DF) & Total TF:")
    print(top_df_terms.to_string())

    # 3. Tampilkan Index TF-IDF yang telah di-TRANSPOSE
    #    (Baris = Term/Token, Kolom = Kode Dokumen)
    print(f"\n[3] INDEX TF-IDF HASIL TRANSPOSE (Baris = Term/Token, Kolom = Dokumen):")
    print("-" * 80)
    print(
        f"  Dimensi Matriks TF-IDF Awal (Dokumen x Term)      : {engine.tfidf_wm.shape}"
    )
    print(
        f"  Dimensi Index Hasil Transpose (Term x Dokumen)    : {engine.tfidf_wm_transposed.shape}"
    )
    print("\n  Cuplikan Tabel Index TF-IDF Transpose (25 Term Teratas Berdasarkan Bobot Rata-rata):")
    # Ambil 25 term dengan rata-rata TF-IDF tertinggi agar nyaman dilihat di layar
    mean_scores = engine.df_tfidf_transpose.mean(axis=1).sort_values(ascending=False)
    top_25_terms = mean_scores.head(25).index
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.6f}".format)
    print(engine.df_tfidf_transpose.loc[top_25_terms].to_string())

    print("\n  Cuplikan Keseluruhan Index TF-IDF Transpose (Baris Pertama & Terakhir):")
    print(engine.df_tfidf_transpose)

    # 4. Loop Interaktif Input Kueri Pengguna
    print(f"\n{sep}")
    print("[4] PENCARIAN KUERI PENGGUNA (COSINE SIMILARITY)")
    print("    Ketik frasa kueri dalam Bahasa Inggris (misal: 'transformer bert short answer grading')")
    print("    Ketik '!ui' untuk membuka Antarmuka Web (Streamlit), atau 'exit' untuk keluar.")
    print(sep)

    # Jika tidak interaktif (mis. dipanggil dari pengujian otomatis), jalankan 1 contoh kueri
    if not sys.stdin.isatty():
        sample_q = "transformer bert semantic similarity short answer grading"
        print(f"\n[Mode Non-Interaktif] Menjalankan kueri contoh: '{sample_q}'")
        _display_cli_search_result(engine, sample_q)
        return

    while True:
        try:
            user_query = input("\nMasukkan Kueri Pencarian > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTerima kasih telah menggunakan Mesin Pencari ASAG!")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit", "keluar", "q"}:
            print("Terima kasih telah menggunakan Mesin Pencari ASAG!")
            break
        if user_query.lower() == "!ui":
            print("Membuka Antarmuka Mesin Pencari Web (Streamlit)...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
            break

        _display_cli_search_result(engine, user_query)


def _display_cli_search_result(engine: ASAGSearchEngine, query: str) -> None:
    """Menampilkan hasil praproses kueri, bobot TF-IDF kueri, dan ranking Cosine Similarity."""
    res = engine.search(query)
    prep_q = res["preprocessed_query"]

    print("-" * 80)
    print(f"Kueri Asli               : {res['raw_query']}")
    print(f"Normalisasi Kueri        : {prep_q['normalized']}")
    print(f"Tokenisasi & Stopword    : {prep_q['filtered_tokens']}")
    print(f"Hasil Stemming Kueri     : {prep_q['stemmed_tokens']}")
    print("Bobot TF-IDF Kueri       :")
    if res["query_tfidf_weights"]:
        for term, weight in res["query_tfidf_weights"].items():
            status = "" if weight > 0 else " (tidak ada di korpus)"
            print(f"  - {term:<20} : {weight:.6f}{status}")
    else:
        print("  (Tidak ada token valid setelah penghilangan stopword)")

    print("\nHASIL PERINGKAT DOKUMEN BERDASARKAN COSINE SIMILARITY TERTINGGI:")
    print(f"{'Rank':<6} | {'Kode Dokumen':<14} | {'Nilai Kemiripan (Cosine)':<26} | {'Nama Berkas Dokumen'}")
    print("-" * 80)
    for item in res["results"]:
        print(
            f"{item['rank']:<6} | {item['code']:<14} | {item['similarity']:<26.6f} | {item['filename']}"
        )
    print("-" * 80)


# ==============================================================================
# BONUS 2.5%: ANTARMUKA MESIN PENCARI SEDERHANA BERBASIS WEB (STREAMLIT)
# ==============================================================================
def run_streamlit_ui() -> None:
    """Menjalankan Antarmuka Mesin Pencari Sederhana menggunakan Streamlit."""
    import streamlit as st

    st.set_page_config(
        page_title="ASAG Scholar Search Engine - TF-IDF & Cosine Similarity",
        page_icon="🔍",
        layout="wide",
    )

    # Styling CSS agar tampilan bersih dan profesional seperti mesin pencari akademik
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            padding: 1.6rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.2rem;
        }
        .main-header h1 {
            color: white !important;
            margin: 0;
            font-size: 1.85rem;
        }
        .main-header p {
            color: #e0e7ff;
            margin: 0.35rem 0 0 0;
            font-size: 0.98rem;
        }
        .result-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #2563eb;
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.04);
        }
        .doc-badge {
            background-color: #dbeafe;
            color: #1e40af;
            padding: 0.2rem 0.65rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
            margin-right: 0.5rem;
        }
        .score-badge {
            background-color: #dcfce7;
            color: #166534;
            padding: 0.2rem 0.65rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
            float: right;
        }
        .zero-badge {
            background-color: #f1f5f9;
            color: #64748b;
            padding: 0.2rem 0.65rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9rem;
            float: right;
        }
        .snippet-text {
            color: #334155;
            font-size: 0.92rem;
            margin-top: 0.5rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-header">
            <h1>🔍 ASAG Scientific Article Search Engine</h1>
            <p>Tugas Pemrograman Kelompok II NLP (TI6073) — Pembobotan TF-IDF, Index Transpose & Cosine Similarity</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar untuk Konfigurasi & Tambah Dokumen Korpus
    with st.sidebar:
        st.header("⚙️ Pengaturan & Korpus")
        lang_choice = st.selectbox(
            "Bahasa Praproses (Stopwords & Stemmer):",
            options=["english", "indonesian"],
            index=0,
            help="Pilih 'english' untuk PorterStemmer + English Stopwords, atau 'indonesian' untuk Sastrawi + Indonesian Stopwords.",
        )

        st.markdown("---")
        st.subheader("📂 Tambah Dokumen Baru")
        st.caption(
            "Unggah file artikel `.pdf` atau `.txt` tambahan dari anggota kelompok (otomatis diekstrak & diindeks):"
        )
        uploaded_files = st.file_uploader(
            "Pilih file .pdf / .txt",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            os.makedirs(CORPUS_DIR, exist_ok=True)
            saved_count = 0
            for uf in uploaded_files:
                target_path = os.path.join(CORPUS_DIR, uf.name)
                with open(target_path, "wb") as out_f:
                    out_f.write(uf.getbuffer())
                saved_count += 1
            if saved_count > 0:
                st.success(f"{saved_count} berkas berhasil ditambahkan ke korpus!")
                st.cache_resource.clear()

        if st.button("🔄 Muat Ulang Indeks Korpus", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    @st.cache_resource(show_spinner="Memproses korpus, menghitung TF-IDF & membangun indeks...")
    def get_engine(language: str) -> ASAGSearchEngine:
        eng = ASAGSearchEngine(corpus_dir=CORPUS_DIR, language=language)
        eng.build_index()
        return eng

    engine = get_engine(lang_choice)

    # Metrik Ringkasan di Bagian Atas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Dokumen Korpus", f"{len(engine.doc_codes)} Dokumen")
    c2.metric("Total Term Unik (Vocabulary)", f"{len(engine.tfidf_terms):,} Term")
    c3.metric(
        "Dimensi Matriks TF-IDF",
        f"{engine.tfidf_wm.shape[0]} × {engine.tfidf_wm.shape[1]}",
    )
    c4.metric(
        "Dimensi Index Transpose",
        f"{engine.tfidf_wm_transposed.shape[0]} × {engine.tfidf_wm_transposed.shape[1]}",
    )

    # Navigasi Tab
    tab_search, tab_index, tab_tfdf, tab_prep = st.tabs(
        [
            "🔍 Mesin Pencari (Cosine Similarity)",
            "📊 Index TF-IDF Transpose (Term × Dokumen)",
            "📈 Frekuensi Token (TF) & Dokumen (DF)",
            "📝 Korpus & Detail Praproses Teks",
        ]
    )

    # --------------------------------------------------------------------------
    # TAB 1: MESIN PENCARI (INPUT KUERI & RANKING COSINE SIMILARITY)
    # --------------------------------------------------------------------------
    with tab_search:
        st.subheader("Pencarian Kueri Frasa pada Korpus Artikel ASAG")

        # Contoh kueri cepat
        example_queries = [
            "transformer bert roberta semantic similarity short answer grading",
            "word embedding word2vec syntactic analysis dependency parsing",
            "gpt-4 large language model zero-shot reference answer",
            "parallel cnn lstm customized multi-head attention",
            "t5 text-to-text feedback generation high school biology",
        ]
        selected_example = st.selectbox(
            "💡 Pilih contoh kueri cepat (atau ketik kueri sendiri di bawah):",
            options=["-- Ketik kueri sendiri --"] + example_queries,
            index=0,
        )

        default_q = (
            selected_example
            if selected_example != "-- Ketik kueri sendiri --"
            else "transformer bert semantic similarity short answer grading"
        )

        col_q, col_btn = st.columns([5, 1])
        with col_q:
            query_input = st.text_input(
                "Masukkan Kueri Frasa Pencarian:",
                value=default_q,
                placeholder="Ketik kata kunci atau frasa pencarian...",
            )
        with col_btn:
            st.write("")
            st.write("")
            do_search = st.button("🔍 Cari Dokumen", type="primary", use_container_width=True)

        if query_input.strip():
            search_res = engine.search(query_input)
            prep_q = search_res["preprocessed_query"]

            # Expander untuk melihat detail praproses kueri & bobot TF-IDF kueri
            with st.expander("🔬 Lihat Detail Praproses Kueri & Bobot TF-IDF Kueri", expanded=True):
                qc1, qc2 = st.columns([3, 2])
                with qc1:
                    st.markdown(f"- **Kueri Asli:** `{search_res['raw_query']}`")
                    st.markdown(f"- **Normalisasi (Case Folding & RegEx):** `{prep_q['normalized']}`")
                    st.markdown(
                        f"- **Tokenisasi & Stopword Removal:** `{prep_q['filtered_tokens']}`"
                    )
                    st.markdown(f"- **Hasil Stemming Kueri:** `{prep_q['stemmed_tokens']}`")
                with qc2:
                    st.markdown("**Bobot TF-IDF Tiap Term pada Kueri:**")
                    if search_res["query_tfidf_weights"]:
                        q_df = pd.DataFrame(
                            [
                                {
                                    "Term Kueri (Stemmed)": k,
                                    "Bobot TF-IDF Kueri": v,
                                    "DF di Korpus": engine.doc_freq.get(k, 0),
                                }
                                for k, v in search_res["query_tfidf_weights"].items()
                            ]
                        )
                        st.dataframe(q_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Tidak ada token kueri tersisa setelah stopword removal.")

            st.markdown("### 🏆 Peringkat Dokumen Berdasarkan Nilai Kemiripan Cosine Tertinggi")

            # Tabel Ringkas Kode Dokumen & Nilai Kemiripan (sesuai instruksi tugas)
            summary_rank_df = pd.DataFrame(
                [
                    {
                        "Peringkat": item["rank"],
                        "Kode Dokumen": item["code"],
                        "Nilai Kemiripan (Cosine Similarity)": round(item["similarity"], 6),
                        "Persentase Kemiripan": f"{item['similarity'] * 100:.2f}%",
                        "Nama Berkas Dokumen": item["filename"],
                    }
                    for item in search_res["results"]
                ]
            )
            st.dataframe(summary_rank_df, use_container_width=True, hide_index=True)

            # Kartu Hasil Mesin Pencari
            st.markdown("#### 📄 Detail Hasil Pencarian Dokumen")
            for item in search_res["results"]:
                sim_val = item["similarity"]
                badge_class = "score-badge" if sim_val > 0 else "zero-badge"
                matched_list = ", ".join(
                    [
                        f"<b>{t}</b> (TF={info['tf_in_doc']}, TF-IDF={info['tfidf_in_doc']:.4f})"
                        for t, info in item["matched_terms"].items()
                    ]
                )
                if not matched_list:
                    matched_list = "<i>Tidak ada term kueri yang cocok</i>"

                st.markdown(
                    f"""
                    <div class="result-card">
                        <span class="doc-badge">#{item['rank']} — Kode: {item['code']}</span>
                        <span class="{badge_class}">Cosine Similarity: {sim_val:.6f} ({sim_val*100:.2f}%)</span>
                        <h4 style="margin: 0.4rem 0 0.2rem 0; color: #0f172a;">{item['title']}</h4>
                        <div style="font-size: 0.82rem; color: #64748b;">Berkas: <code>{item['filename']}</code> | Term Kueri yang Cocok: {matched_list}</div>
                        <div class="snippet-text">{item['snippet']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------------------------
    # TAB 2: INDEX TF-IDF TRANSPOSE (TERM SEBAGAI BARIS, DOKUMEN SEBAGAI KOLOM)
    # --------------------------------------------------------------------------
    with tab_index:
        st.subheader("Tabel Index TF-IDF Hasil Transpose (`tfidf_wm.transpose()`)")
        st.info(
            "Sesuai instruksi tugas: Matriks TF-IDF telah diubah menggunakan fungsi `.transpose()` "
            "sehingga **nama/kode dokumen menjadi kolom (`D01`, `D02`, ...)** dan **term/token menjadi baris**."
        )

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            term_filter = st.text_input(
                "🔎 Filter / Cari Term di dalam Index:",
                value="",
                placeholder="Ketik awalan atau bagian term (misal: grad, bert, semant)...",
            )
        with col_f2:
            sort_by_col = st.selectbox(
                "Urutkan Baris Berdasarkan:",
                options=["Abjad Term (A-Z)", "Rata-rata TF-IDF Tertinggi"] + engine.doc_codes,
                index=1,
            )

        df_view = engine.df_tfidf_transpose.copy()
        if term_filter.strip():
            df_view = df_view[df_view.index.str.contains(term_filter.strip().lower(), na=False)]

        if sort_by_col == "Rata-rata TF-IDF Tertinggi":
            df_view = df_view.loc[df_view.mean(axis=1).sort_values(ascending=False).index]
        elif sort_by_col in engine.doc_codes:
            df_view = df_view.sort_values(by=sort_by_col, ascending=False)

        st.dataframe(
            df_view.style.format("{:.6f}"),
            use_container_width=True,
            height=500,
        )

        with st.expander("Lihat Matriks TF-IDF Sebelum Transpose (Baris = Dokumen, Kolom = Term)"):
            st.dataframe(
                engine.df_tfidf_docs.style.format("{:.6f}"),
                use_container_width=True,
            )

    # --------------------------------------------------------------------------
    # TAB 3: FREKUENSI TOKEN (TF) & DOCUMENT FREQUENCY (DF)
    # --------------------------------------------------------------------------
    with tab_tfdf:
        st.subheader("Frekuensi Tiap Token di Tiap Dokumen (TF) & Document Frequency (DF)")
        tf_df_table = engine.get_tf_df_summary_table()

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            st.markdown("**Tabel Frekuensi Token (TF per Dokumen, Total TF, dan Document Frequency / DF):**")
            st.dataframe(
                tf_df_table.sort_values(by=["DF (Doc Freq)", "Total_TF"], ascending=False),
                use_container_width=True,
                height=460,
            )
        with col_s2:
            selected_doc_chart = st.selectbox(
                "Pilih Dokumen untuk Grafik Top 20 Term (FreqDist):",
                options=engine.doc_codes,
            )
            top_20 = engine.doks_tf[selected_doc_chart].most_common(20)
            chart_df = pd.DataFrame(top_20, columns=["Term", "Frekuensi (TF)"]).set_index("Term")
            st.markdown(f"**Distribusi 20 Term Teratas pada `{selected_doc_chart}`:**")
            st.bar_chart(chart_df)

    # --------------------------------------------------------------------------
    # TAB 4: KORPUS & DETAIL PRAPROSES TEKS
    # --------------------------------------------------------------------------
    with tab_prep:
        st.subheader("Daftar Dokumen Korpus & Hasil Tiap Tahap Praproses")
        selected_doc = st.selectbox(
            "Pilih Kode Dokumen untuk Melihat Hasil Praproses:",
            options=engine.doc_codes,
            format_func=lambda c: f"{c} — {engine.documents[c]['filename']}",
        )
        doc_data = engine.documents[selected_doc]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah Karakter Asli", f"{len(doc_data['raw_text']):,}")
        m2.metric("Jumlah Token Awal", f"{len(doc_data['tokens']):,}")
        m3.metric("Setelah Stopword Removal", f"{len(doc_data['filtered_tokens']):,}")
        m4.metric("Term Unik (Setelah Stemming)", f"{len(engine.doks_tf[selected_doc]):,}")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("**1. Cuplikan Teks Asli (Raw Text):**")
            st.text_area(
                "Raw Text",
                value=doc_data["raw_text"][:2500],
                height=220,
                disabled=True,
                label_visibility="collapsed",
            )
            st.markdown("**2. Hasil Tokenisasi (`nltk.word_tokenize`) - 100 Token Pertama:**")
            st.code(str(doc_data["tokens"][:100]))
        with p_col2:
            st.markdown("**3. Setelah Penghilangan Stopword (`nltk.corpus.stopwords`) - 100 Token Pertama:**")
            st.code(str(doc_data["filtered_tokens"][:100]))
            st.markdown("**4. Hasil Akhir Setelah Stemming (`PorterStemmer`) - Cuplikan:**")
            st.text_area(
                "Stemmed Text",
                value=doc_data["clean_text"][:2500],
                height=220,
                disabled=True,
                label_visibility="collapsed",
            )


# ==============================================================================
# ENTRY POINT UTAMA (.PY)
# ==============================================================================
def _is_running_in_streamlit() -> bool:
    """Mendeteksi apakah skrip sedang dijalankan melalui `streamlit run` tanpa memicu warning di CLI."""
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__":
    # Pastikan stdout mendukung karakter UTF-8 di terminal Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if _is_running_in_streamlit():
        run_streamlit_ui()
    elif "--ui" in sys.argv:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        run_cli_mode()
elif _is_running_in_streamlit():
    run_streamlit_ui()
