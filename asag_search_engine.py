<<<<<<< Updated upstream
# Tugas 2 NLP - Search Engine ASAG
=======
"""
TUGAS PEMROGRAMAN KELOMPOK II - NATURAL LANGUAGE PROCESSING (TI6073)
Topik: Automated Short Answer Grading (ASAG)

Alur Program Sesuai Ketentuan Tugas:
1. Membaca korpus dokumen teks.
2. Praproses: Normalisasi (case folding, hapus tanda baca & angka),
   Tokenisasi (NLTK), Stopword Removal (NLTK), dan Stemming (PorterStemmer).
3. Menghitung Term Frequency (TF) tiap dokumen dan Document Frequency (DF).
4. Menghitung TF-IDF (sklearn), menyimpan dalam index, dan melakukan
   Transpose (.transpose()) agar Dokumen menjadi Kolom dan Term menjadi Baris.
5. Menerima input kueri pengguna, praproses kueri, hitung bobot TF-IDF kueri,
   hitung Cosine Similarity, dan tampilkan peringkat dokumen tertinggi.
6. BONUS 2.5%: Antarmuka Mesin Pencari Sederhana (Streamlit).
"""
>>>>>>> Stashed changes

import os
import re
import sys
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

<<<<<<< Updated upstream
# download resource nltk jika belum ada
=======
# Unduh paket NLTK jika belum ada
>>>>>>> Stashed changes
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
PDF_DIR = os.path.join(BASE_DIR, "PDF")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


<<<<<<< Updated upstream
# 1. baca file korpus (ekstrak pdf jika teks belum ada)
def load_corpus() -> dict:
    os.makedirs(CORPUS_DIR, exist_ok=True)

=======
# ==============================================================================
# 1. BACA DOKUMEN & EKSTRAKSI TEKS
# ==============================================================================
def load_corpus() -> dict:
    """Membaca dokumen teks dari folder corpus/ (atau ekstrak dari PDF jika perlu)."""
    os.makedirs(CORPUS_DIR, exist_ok=True)

    # Ekstrak otomatis dari PDF jika di corpus/ belum ada file .txt
>>>>>>> Stashed changes
    if os.path.exists(PDF_DIR):
        for root, _, files in os.walk(PDF_DIR):
            for file in files:
                if file.lower().endswith(".pdf"):
                    base_name = os.path.splitext(file)[0]
                    txt_path = os.path.join(CORPUS_DIR, f"{base_name}.txt")
                    if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                        try:
                            import fitz
                            doc = fitz.open(os.path.join(root, file))
                            text = "\n".join([page.get_text() for page in doc])
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(text)
                        except Exception:
                            pass

    txt_files = sorted([f for f in os.listdir(CORPUS_DIR) if f.lower().endswith(".txt")])
    corpus = {}
    for idx, fname in enumerate(txt_files, start=1):
        doc_code = f"D{idx:02d}"
        with open(os.path.join(CORPUS_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
            corpus[doc_code] = {"filename": fname, "text": f.read()}
    return corpus


<<<<<<< Updated upstream
# 2. praproses: lowercase, regex, tokenisasi, stopword, stemming
def preprocess_text(text: str) -> list:
=======
# ==============================================================================
# 2. PRAPROSES TEKS (NORMALISASI, TOKENISASI, STOPWORD, STEMMING)
# ==============================================================================
def preprocess_text(text: str) -> list:
    """Melakukan normalisasi, tokenisasi, penghapusan stopword, dan stemming."""
>>>>>>> Stashed changes
    text = text.lower()
    text = re.sub(r"[\W\s0-9_]+", " ", text)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 1 and w.isascii()]
    stemmed = [stemmer.stem(w) for w in filtered]
    return stemmed


<<<<<<< Updated upstream
# 3. hitung tf, df, tf-idf, dan transpose matriks index
=======
# ==============================================================================
# 3. PEMBUATAN INDEKS TF-IDF & TRANSPOSE
# ==============================================================================
>>>>>>> Stashed changes
def build_index(corpus: dict):
    doc_codes = list(corpus.keys())
    clean_texts = []
    doc_tfs = {}
    df_counter = Counter()

    for code in doc_codes:
        tokens = preprocess_text(corpus[code]["text"])
        corpus[code]["tokens"] = tokens
        clean_texts.append(" ".join(tokens))
        
<<<<<<< Updated upstream
        # frekuensi token per dokumen (TF)
        doc_tfs[code] = nltk.FreqDist(tokens)
        
        # document frequency (DF)
        for term in set(tokens):
            df_counter[term] += 1

    # pembobotan tf-idf
=======
        # Hitung frekuensi tiap token di tiap dokumen (Term Frequency)
        doc_tfs[code] = nltk.FreqDist(tokens)
        
        # Hitung Document Frequency (DF)
        for term in set(tokens):
            df_counter[term] += 1

    # Pembobotan TF-IDF menggunakan sklearn
>>>>>>> Stashed changes
    tfidf_vec = TfidfVectorizer()
    tfidf_wm = tfidf_vec.fit_transform(clean_texts)
    terms = tfidf_vec.get_feature_names_out()

<<<<<<< Updated upstream
    # transpose: term jadi baris, dokumen jadi kolom
=======
    # Transpose matriks agar Dokumen menjadi KOLOM dan Term menjadi BARIS
>>>>>>> Stashed changes
    tfidf_transposed = tfidf_wm.transpose()
    df_index_transposed = pd.DataFrame(
        data=tfidf_transposed.toarray(),
        index=terms,
        columns=doc_codes
    )
    df_index_transposed.index.name = "Term / Token"

    return doc_codes, terms, doc_tfs, df_counter, tfidf_vec, tfidf_wm, df_index_transposed


<<<<<<< Updated upstream
# 4. pencarian kueri dengan cosine similarity
def search_query(query: str, tfidf_vec, tfidf_wm, doc_codes, corpus):
    query_tokens = preprocess_text(query)
    clean_query = " ".join(query_tokens)

    # vektor tf-idf kueri
    query_vec = tfidf_vec.transform([clean_query])

    # hitung kemiripan cosine
    sim_scores = cosine_similarity(query_vec, tfidf_wm)[0]

=======
# ==============================================================================
# 4. PENCARIAN KUERI & COSINE SIMILARITY
# ==============================================================================
def search_query(query: str, tfidf_vec, tfidf_wm, doc_codes, corpus):
    """Memproses kueri frasa dan menghitung Cosine Similarity dengan dokumen."""
    query_tokens = preprocess_text(query)
    clean_query = " ".join(query_tokens)

    # Bobot TF-IDF kueri
    query_vec = tfidf_vec.transform([clean_query])

    # Hitung Cosine Similarity
    sim_scores = cosine_similarity(query_vec, tfidf_wm)[0]

    # Urutkan dokumen dari kemiripan tertinggi
>>>>>>> Stashed changes
    results = []
    for idx, code in enumerate(doc_codes):
        results.append({
            "code": code,
            "filename": corpus[code]["filename"],
            "similarity": float(sim_scores[idx])
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return query_tokens, results


<<<<<<< Updated upstream
# mode terminal (cli)
def run_cli():
    print("=" * 60)
    print("Mesin Pencari Dokumen ASAG (TF-IDF & Cosine Similarity)")
    print("=" * 60)

    corpus = load_corpus()
    if not corpus:
        print("Tidak ada dokumen di folder corpus/.")
=======
# ==============================================================================
# 5. MODE TERMINAL (CLI)
# ==============================================================================
def run_cli():
    print("=" * 70)
    print("  MESIN PENCARI ARTIKEL ASAG (TF-IDF & COSINE SIMILARITY)")
    print("=" * 70)

    corpus = load_corpus()
    if not corpus:
        print("[Error] Tidak ada dokumen ditemukan di folder corpus/.")
>>>>>>> Stashed changes
        return

    doc_codes, terms, doc_tfs, df_counter, tfidf_vec, tfidf_wm, df_transposed = build_index(corpus)

<<<<<<< Updated upstream
    print(f"\nKorpus terbaca: {len(doc_codes)} dokumen")
    for code in doc_codes:
        top3 = doc_tfs[code].most_common(3)
        print(f"  {code} - {corpus[code]['filename'][:45]}... | Top TF: {top3}")

    print("\nContoh Document Frequency (DF) 10 term teratas:")
    top_df = sorted(df_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    for term, freq in top_df:
        print(f"  - {term:<12}: {freq} dokumen")

    print("\nCuplikan indeks TF-IDF (transpose):")
    print(df_transposed.head(10))

    print("\n" + "=" * 60)
    print("Pencarian kueri (ketik 'exit' untuk keluar, atau '!ui' untuk web)")
    print("=" * 60)

    if not sys.stdin.isatty():
        sample = "semantic similarity automated short answer grading"
        print(f"\nKueri: {sample}")
        q_tokens, results = search_query(sample, tfidf_vec, tfidf_wm, doc_codes, corpus)
        print("Hasil:")
        for r in results:
            print(f"  {r['code']} | Sim: {r['similarity']:.6f} | {r['filename']}")
=======
    print(f"\n[1] Korpus Terbaca: {len(doc_codes)} Dokumen ({', '.join(doc_codes)})")
    for code in doc_codes:
        top3 = doc_tfs[code].most_common(3)
        print(f"    {code} - {corpus[code]['filename'][:50]}... | Top TF: {top3}")

    print("\n[2] Contoh Document Frequency (DF) untuk 10 Term Teratas:")
    top_df = sorted(df_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    for term, freq in top_df:
        print(f"    - {term:<15}: {freq} dokumen")

    print("\n[3] Cuplikan Indeks TF-IDF Hasil Transpose (Term = Baris, Dokumen = Kolom):")
    print(df_transposed.head(10))

    print("\n" + "=" * 70)
    print("PENCARIAN KUERI (Ketik 'exit' untuk keluar, atau '!ui' untuk buka Web UI)")
    print("=" * 70)

    # Jika berjalan non-interaktif
    if not sys.stdin.isatty():
        sample = "semantic similarity automated short answer grading"
        print(f"\n[Kueri Contoh]: {sample}")
        q_tokens, results = search_query(sample, tfidf_vec, tfidf_wm, doc_codes, corpus)
        print("Peringkat Kemiripan Dokumen:")
        for r in results:
            print(f"  {r['code']} | Nilai Kemiripan: {r['similarity']:.6f} | {r['filename']}")
>>>>>>> Stashed changes
        return

    while True:
        try:
<<<<<<< Updated upstream
            q = input("\nMasukkan kueri: ").strip()
=======
            q = input("\nMasukkan kueri pencarian: ").strip()
>>>>>>> Stashed changes
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ["exit", "quit", "keluar"]:
            break
        if q.lower() == "!ui":
            os.system(f"streamlit run \"{__file__}\"")
            break

        q_tokens, results = search_query(q, tfidf_vec, tfidf_wm, doc_codes, corpus)
<<<<<<< Updated upstream
        print(f"Token kueri: {q_tokens}")
        print("\nHasil pencarian:")
        print(f"{'Kode':<6} | {'Cosine Sim':<12} | Nama File")
        print("-" * 60)
        for r in results:
            print(f"{r['code']:<6} | {r['similarity']:<12.6f} | {r['filename']}")


# mode web (streamlit)
def run_web_ui():
    import streamlit as st

    st.set_page_config(page_title="Mesin Pencari ASAG", layout="centered")
    st.title("Mesin Pencari Dokumen ASAG")
    st.caption("Tugas Pemrograman NLP - TF-IDF & Cosine Similarity")
=======
        print(f"Token kueri setelah praproses: {q_tokens}")
        print("\nHasil Urutan Kemiripan (Cosine Similarity):")
        print(f"{'Kode Dokumen':<15} | {'Nilai Kemiripan':<18} | Nama Berkas")
        print("-" * 70)
        for r in results:
            print(f"{r['code']:<15} | {r['similarity']:<18.6f} | {r['filename']}")


# ==============================================================================
# 6. BONUS 2.5%: ANTARMUKA MESIN PENCARI SEDERHANA (STREAMLIT)
# ==============================================================================
def run_web_ui():
    import streamlit as st

    st.set_page_config(page_title="Mesin Pencari Dokumen ASAG", layout="centered")
    st.title("🔍 Mesin Pencari Dokumen Ilmiah ASAG")
    st.caption("Tugas Pemrograman NLP Kelompok II — TF-IDF & Cosine Similarity")
>>>>>>> Stashed changes

    corpus = load_corpus()
    if not corpus:
        st.error("Tidak ada dokumen ditemukan di folder corpus/.")
        return

    doc_codes, terms, doc_tfs, df_counter, tfidf_vec, tfidf_wm, df_transposed = build_index(corpus)

<<<<<<< Updated upstream
    st.write(f"Total Dokumen: {len(doc_codes)} | Total Vocabulary: {len(terms)}")

    query = st.text_input("Masukkan kueri pencarian:", value="semantic similarity short answer grading")
    
    if st.button("Cari", type="primary"):
        if query.strip():
            q_tokens, results = search_query(query, tfidf_vec, tfidf_wm, doc_codes, corpus)
            st.info(f"Token kueri: `{q_tokens}`")
=======
    st.write(f"**Korpus Terindeks:** {len(doc_codes)} Dokumen | **Total Vocabulary:** {len(terms)} Kata Unik")

    query = st.text_input("Ketikkan kueri pencarian Anda:", value="semantic similarity short answer grading")
    
    if st.button("Cari Dokumen", type="primary"):
        if query.strip():
            q_tokens, results = search_query(query, tfidf_vec, tfidf_wm, doc_codes, corpus)
            st.info(f"**Token kueri setelah praproses:** `{q_tokens}`")
>>>>>>> Stashed changes
            
            st.subheader("Hasil Peringkat Dokumen:")
            for rank, r in enumerate(results, start=1):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(f"#{rank} {r['code']}", f"{r['similarity']:.4f}")
                with col2:
                    st.write(f"**{r['filename']}**")
<<<<<<< Updated upstream
                    st.caption(f"Cosine Similarity: {r['similarity']:.6f}")
                st.divider()

    with st.expander("Lihat Indeks TF-IDF Transpose"):
        st.dataframe(df_transposed)


# deteksi runtime streamlit
def is_streamlit_running() -> bool:
=======
                    st.caption(f"Nilai Kemiripan Cosine: {r['similarity']:.6f}")
                st.divider()

    with st.expander("Lihat Indeks TF-IDF Transpose (Term sebagai Baris, Dokumen sebagai Kolom)"):
        st.dataframe(df_transposed)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def is_streamlit_running() -> bool:
    """Mendeteksi apakah skrip sedang dieksekusi oleh runtime Streamlit."""
>>>>>>> Stashed changes
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if is_streamlit_running():
    run_web_ui()
elif __name__ == "__main__":
    if "--ui" in sys.argv:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        run_cli()
