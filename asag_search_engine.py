# Tugas 2 NLP - Search Engine ASAG

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

# download resource nltk jika belum ada
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


# 1. baca file korpus (ekstrak pdf jika teks belum ada)
def load_corpus() -> dict:
    os.makedirs(CORPUS_DIR, exist_ok=True)

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


# 2. praproses: lowercase, regex, tokenisasi, stopword, stemming
def preprocess_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[\W\s0-9_]+", " ", text)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 1 and w.isascii()]
    stemmed = [stemmer.stem(w) for w in filtered]
    return stemmed


# 3. hitung tf, df, tf-idf, dan transpose matriks index
def build_index(corpus: dict):
    doc_codes = list(corpus.keys())
    clean_texts = []
    doc_tfs = {}
    df_counter = Counter()

    for code in doc_codes:
        tokens = preprocess_text(corpus[code]["text"])
        corpus[code]["tokens"] = tokens
        clean_texts.append(" ".join(tokens))
        
        # frekuensi token per dokumen (TF)
        doc_tfs[code] = nltk.FreqDist(tokens)
        
        # document frequency (DF)
        for term in set(tokens):
            df_counter[term] += 1

    # pembobotan tf-idf
    tfidf_vec = TfidfVectorizer()
    tfidf_wm = tfidf_vec.fit_transform(clean_texts)
    terms = tfidf_vec.get_feature_names_out()

    # transpose: term jadi baris, dokumen jadi kolom
    tfidf_transposed = tfidf_wm.transpose()
    df_index_transposed = pd.DataFrame(
        data=tfidf_transposed.toarray(),
        index=terms,
        columns=doc_codes
    )
    df_index_transposed.index.name = "Term / Token"

    return doc_codes, terms, doc_tfs, df_counter, tfidf_vec, tfidf_wm, df_index_transposed


# 4. pencarian kueri dengan cosine similarity
def search_query(query: str, tfidf_vec, tfidf_wm, doc_codes, corpus):
    query_tokens = preprocess_text(query)
    clean_query = " ".join(query_tokens)

    # vektor tf-idf kueri
    query_vec = tfidf_vec.transform([clean_query])

    # hitung kemiripan cosine
    sim_scores = cosine_similarity(query_vec, tfidf_wm)[0]

    results = []
    for idx, code in enumerate(doc_codes):
        results.append({
            "code": code,
            "filename": corpus[code]["filename"],
            "similarity": float(sim_scores[idx])
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return query_tokens, results


# mode terminal (cli)
def run_cli():
    print("=" * 60)
    print("Mesin Pencari Dokumen ASAG (TF-IDF & Cosine Similarity)")
    print("=" * 60)

    corpus = load_corpus()
    if not corpus:
        print("Tidak ada dokumen di folder corpus/.")
        return

    doc_codes, terms, doc_tfs, df_counter, tfidf_vec, tfidf_wm, df_transposed = build_index(corpus)

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
        return

    while True:
        try:
            q = input("\nMasukkan kueri: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ["exit", "quit", "keluar"]:
            break
        if q.lower() == "!ui":
            os.system(f"streamlit run \"{__file__}\"")
            break

        q_tokens, results = search_query(q, tfidf_vec, tfidf_wm, doc_codes, corpus)
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

    corpus = load_corpus()
    if not corpus:
        st.error("Tidak ada dokumen ditemukan di folder corpus/.")
        return

    doc_codes, terms, doc_tfs, df_counter, tfidf_vec, tfidf_wm, df_transposed = build_index(corpus)

    st.write(f"Total Dokumen: {len(doc_codes)} | Total Vocabulary: {len(terms)}")

    query = st.text_input("Masukkan kueri pencarian:", value="semantic similarity short answer grading")
    
    if st.button("Cari", type="primary"):
        if query.strip():
            q_tokens, results = search_query(query, tfidf_vec, tfidf_wm, doc_codes, corpus)
            st.info(f"Token kueri: `{q_tokens}`")
            
            st.subheader("Hasil Peringkat Dokumen:")
            for rank, r in enumerate(results, start=1):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(f"#{rank} {r['code']}", f"{r['similarity']:.4f}")
                with col2:
                    st.write(f"**{r['filename']}**")
                    st.caption(f"Cosine Similarity: {r['similarity']:.6f}")
                st.divider()

    with st.expander("Lihat Indeks TF-IDF Transpose"):
        st.dataframe(df_transposed)


# deteksi runtime streamlit
def is_streamlit_running() -> bool:
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
