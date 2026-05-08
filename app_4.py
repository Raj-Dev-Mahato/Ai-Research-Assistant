import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
import faiss
import json

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from groq import Groq

# -------------------------------
# 🔹 Streamlit Config (only once)
# -------------------------------
st.set_page_config(
    page_title="Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 🔹 CACHED MODELS
# -------------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embed_model, reranker

embedding_model, reranker = load_models()




# -------------------------------
# 🔹 PDF LOADER
# -------------------------------
def load_pdf(file):
    documents = []
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for i, page in enumerate(doc):
        text = page.get_text()
        if text and len(text.strip()) > 50:
            documents.append({"text": text, "page": i + 1, "source": file.name})
    return documents


# -------------------------------
# 🔹 CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\[[0-9,\s]+\]', '', text)
    text = re.sub(r'\s*\.{3,}\s*\d+', '', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line.split()) < 5:
            continue
        if any(x in line.lower() for x in ['acknowledgment', 'acknowledgement', 'funding source', 'conflict of interest', 'declaration of']):
            continue
        cleaned.append(line)
    return ' '.join(cleaned)


# -------------------------------
# 🔹 CHUNKING
# -------------------------------
def is_bad_chunk(text):
    text = text.lower()
    bad_patterns = ["references", "acknowledg", "appendix", "figure", "table", "copyright", "arxiv"]
    return any(p in text for p in bad_patterns)


def semantic_chunk(text, model, threshold=0.45, max_sentences=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    if len(sentences) < 2:
        return sentences if sentences else []
    embeddings = model.encode(sentences, show_progress_bar=False)
    chunks = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [embeddings[i - 1]])[0][0]
        if sim < threshold or len(current_group) >= max_sentences:
            chunk = " ".join(current_group)
            if not is_bad_chunk(chunk):
                chunks.append(chunk)
            current_group = []
        current_group.append(sentences[i])
    if current_group:
        chunk = " ".join(current_group)
        if not is_bad_chunk(chunk):
            chunks.append(chunk)
    return chunks


# -------------------------------
# 🔹 PROCESS DOCUMENTS
# -------------------------------
def process_documents(documents, embed_model):
    chunks = []
    for doc in documents:
        clean_doc = clean_text(doc['text'])
        semantic_chunks = semantic_chunk(clean_doc, embed_model)
        for chunk_text in semantic_chunks:
            chunks.append({'chunk': chunk_text, 'page': doc['page'], 'source': doc['source']})
    return chunks


# -------------------------------
# 🔹 BUILD PIPELINE
# -------------------------------
def build_pipeline(documents):
    chunks = []
    for doc in documents:
        clean_doc = clean_text(doc['text'])
        semantic_chunks = semantic_chunk(clean_doc, embedding_model)
        for chunk_text in semantic_chunks:
            chunks.append({"chunk": chunk_text, "page": doc["page"]})
    texts = [c["chunk"] for c in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return chunks, index, bm25


# -------------------------------
# 🔹 STOPWORDS
# -------------------------------
STOPWORDS = {
    'what', 'is', 'are', 'how', 'does', 'do', 'the', 'a', 'an',
    'in', 'of', 'to', 'for', 'and', 'or', 'with', 'by', 'from',
    'this', 'that', 'which', 'can', 'could', 'would', 'between',
    'used', 'use', 'using', 'describe', 'explain', 'why', 'when',
    'who', 'where', 'was', 'were', 'has', 'have', 'had', 'been',
    'be', 'its', 'their', 'during', 'into', 'about', 'each'
}


# -------------------------------
# 🔹 QUERY KEYWORD EXTRACTION
# -------------------------------
def extract_query_keywords(query, top_n=6):
    words = re.findall(r'\b[a-z]{3,}\b', query.lower())
    return [w for w in words if w not in STOPWORDS][:top_n]


# -------------------------------
# 🔹 RECIPROCAL RANK FUSION
# -------------------------------
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, (chunk_id, _) in enumerate(dense_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(sparse_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# -------------------------------
# 🔹 SENTENCE IMPORTANCE SCORER
# -------------------------------
def score_sentence_importance(sentence):
    s = sentence.lower().strip()
    score = 0.0
    if re.search(r'\b(is defined as|refers to|is a type of|is a method|is a model|denotes|represents)\b', s):
        score += 2.5
    if re.search(r'\b(we propose|we present|we introduce|this paper|our approach|we describe)\b', s):
        score += 2.0
    if re.search(r'\b(we find|we show|results show|outperforms|achieves|state[- ]of[- ]the[- ]art|demonstrate)\b', s):
        score += 1.5
    if re.search(r'\b(dataset|benchmark|evaluated on|tested on|trained on|corpus|test set)\b', s):
        score += 1.0
    if re.search(r'\b(however|unlike|in contrast|compared to|whereas|limitation)\b', s):
        score += 0.5
    if len(s.split()) < 8:
        score -= 1.0
    if re.search(r'\[\d+\]|et al|arxiv\.org|doi\.org', s):
        score -= 2.0
    if len(s.split()) > 40:
        score -= 0.5
    return score


def extract_top_sentences(retrieved_chunks, query, n=6):
    keywords = extract_query_keywords(query)
    scored = []
    for chunk in retrieved_chunks:
        for sent in re.split(r'(?<=[.!?])\s+', chunk['chunk']):
            sent = sent.strip()
            if len(sent.split()) < 5:
                continue
            s = score_sentence_importance(sent)
            overlap = sum(1 for kw in keywords if kw in sent.lower())
            s += overlap * 1.5
            if overlap >= 2:
                s += 1.5
            scored.append((s, sent, chunk.get('page', '?')))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:n]


# -------------------------------
# 🔹 HYBRID SEARCH WITH RRF
# -------------------------------
def hybrid_search_rrf(query, chunks, index, bm25, top_k=10):
    q_emb = embedding_model.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype='float32')
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, min(top_k * 3, len(chunks)))
    dense_results = [(int(idxs[0][i]), float(scores[0][i])) for i in range(len(idxs[0])) if idxs[0][i] >= 0]
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 3]
    sparse_results = [(i, float(bm25_scores[i])) for i in sparse_top]
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    seen, selected = set(), []
    for chunk_id, _ in fused:
        if chunk_id >= len(chunks):
            continue
        text_prefix = chunks[chunk_id]['chunk'][:100]
        if text_prefix not in seen:
            seen.add(text_prefix)
            selected.append(chunks[chunk_id])
        if len(selected) == top_k:
            break
    return selected


def rerank(query, candidate_chunks, top_n=5):
    if not candidate_chunks:
        return []
    pairs = [(query, c['chunk']) for c in candidate_chunks]
    ce_scores = reranker.predict(pairs)
    ranked = sorted(zip(ce_scores, candidate_chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_n]]


# -------------------------------
# 🔹 SEARCH
# -------------------------------
def search(query, chunks, index, bm25, top_k=5):
    q_emb = embedding_model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)
    _, dense_idxs = index.search(q_emb, top_k)
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_idxs = np.argsort(bm25_scores)[::-1][:top_k]
    combined_ids = list(set(dense_idxs[0]) | set(sparse_idxs))
    candidates = [chunks[i] for i in combined_ids if i < len(chunks)]
    pairs = [(query, c["chunk"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# -------------------------------
# 🔹 ANSWER GENERATION
# -------------------------------
def generate_answer(query, chunks, index, bm25, top_k=5):
    candidates = hybrid_search_rrf(query, chunks, index, bm25, top_k=top_k * 2)
    results = rerank(query, candidates, top_n=top_k)
    context = "\n".join([c["chunk"] for c in results[:3]])
    answer = f"{context[:500]}..."
    return answer, results


# -------------------------------
# 🔹 QUERY EXPANSION
# -------------------------------
def expand_query(query, n_variants=3):
    prompt = (
        f'Generate {n_variants} alternative phrasings of this question that mean the same thing.\n'
        f'Output ONLY the alternatives, one per line. No numbering, no extra text.\n\nQuestion: {query}'
    )
    try:
        resp = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_tokens=150
        )
        variants = resp.choices[0].message.content.strip().split('\n')
        variants = [v.strip() for v in variants if v.strip()]
        return [query] + variants[:n_variants]
    except Exception:
        return [query]


# -------------------------------
# 🔹 CONFIDENCE
# -------------------------------
def compute_confidence(top_sentences):
    if not top_sentences:
        return 0
    scores = [s[0] for s in top_sentences[:5]]
    weights = [1.0, 0.9, 0.8, 0.7, 0.6][:len(scores)]
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    max_score = max(scores) if max(scores) > 0 else 1
    confidence = weighted_score / max_score
    confidence *= min(1.0, max_score / 3)
    confidence = max(0.0, min(1.0, confidence))
    return round(confidence * 100, 2)


# ═══════════════════════════════════════════════════════════════
#  MODE CONFIG — colours + personality per search mode
# ═══════════════════════════════════════════════════════════════
MODE_CONFIG = {
    " Focused": {
        "top_k": 3,
        "desc": "Laser-precise answers. Minimal context, maximum signal.",
        "accent":        "#2563eb",   # blue-600
        "accent2":       "#60a5fa",   # blue-400
        "glow":          "rgba(37, 99, 235, 0.08)",
        "glow_strong":   "rgba(37, 99, 235, 0.18)",
        "orb1":          "rgba(96, 165, 250, 0.12)",
        "orb2":          "rgba(37, 99, 235, 0.07)",
        "bg_tint":       "rgba(239, 246, 255, 0.8)",
        "label":         "FOCUSED",
        "tag_color":     "#dbeafe",
        "tag_text":      "#1d4ed8",
    },
    " Balanced": {
        "top_k": 5,
        "desc": "Recommended. Best balance of accuracy and context.",
        "accent":        "#1d4ed8",   # blue-700
        "accent2":       "#3b82f6",   # blue-500
        "glow":          "rgba(29, 78, 216, 0.08)",
        "glow_strong":   "rgba(29, 78, 216, 0.18)",
        "orb1":          "rgba(59, 130, 246, 0.12)",
        "orb2":          "rgba(29, 78, 216, 0.07)",
        "bg_tint":       "rgba(239, 246, 255, 0.8)",
        "label":         "BALANCED",
        "tag_color":     "#dbeafe",
        "tag_text":      "#1e40af",
    },
    " Detailed": {
        "top_k": 7,
        "desc": "Wider context for nuanced, thorough understanding.",
        "accent":        "#0369a1",   # sky-700
        "accent2":       "#38bdf8",   # sky-400
        "glow":          "rgba(3, 105, 161, 0.08)",
        "glow_strong":   "rgba(3, 105, 161, 0.18)",
        "orb1":          "rgba(56, 189, 248, 0.12)",
        "orb2":          "rgba(3, 105, 161, 0.07)",
        "bg_tint":       "rgba(240, 249, 255, 0.8)",
        "label":         "DETAILED",
        "tag_color":     "#e0f2fe",
        "tag_text":      "#0369a1",
    },
    " Deep Dive": {
        "top_k": 10,
        "desc": "Maximum coverage. Full corpus sweep for complex queries.",
        "accent":        "#4f46e5",   # indigo-600
        "accent2":       "#818cf8",   # indigo-400
        "glow":          "rgba(79, 70, 229, 0.08)",
        "glow_strong":   "rgba(79, 70, 229, 0.18)",
        "orb1":          "rgba(129, 140, 248, 0.12)",
        "orb2":          "rgba(79, 70, 229, 0.07)",
        "bg_tint":       "rgba(238, 242, 255, 0.8)",
        "label":         "DEEP DIVE",
        "tag_color":     "#e0e7ff",
        "tag_text":      "#4338ca",
    },
}


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — mode picker (plain, no custom HTML)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✦ Search Mode")
    mode = st.radio(
        "Select mode",
        list(MODE_CONFIG.keys()),
        index=1,
        label_visibility="collapsed"
    )

cfg = MODE_CONFIG[mode]
top_k = cfg["top_k"]

# ═══════════════════════════════════════════════════════════════
#  DYNAMIC CSS — swaps accent colours per mode
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

/* ─── Reset & Base ──────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}

html, body, [data-testid="stAppViewContainer"] {{
    background: #f0f6ff !important;
    color: #1e293b !important;
    font-family: 'DM Sans', sans-serif !important;
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse 80% 50% at 15% -10%, {cfg["orb1"]}, transparent),
        radial-gradient(ellipse 60% 40% at 85% 10%, {cfg["orb2"]}, transparent),
        #f0f6ff !important;
    transition: background 0.8s ease;
}}

/* ─── Sidebar ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: #ffffff !important;
    border-right: 1px solid #dde8f8 !important;
    box-shadow: 2px 0 16px rgba(29, 78, 216, 0.06);
}}

[data-testid="stSidebar"] * {{
    color: #374151 !important;
}}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #1e293b !important;
}}

[data-testid="stSidebar"] .stRadio label {{
    color: #4b5563 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    transition: color 0.2s;
}}

[data-testid="stSidebar"] .stRadio label:hover {{
    color: {cfg["accent"]} !important;
}}

[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {{
    background: {cfg["accent"]} !important;
    box-shadow: 0 0 8px {cfg["glow_strong"]};
}}

/* ─── Hide Streamlit chrome ─────────────────────────────────── */
#MainMenu, footer {{ display: none !important; }}

/* ─── Main padding ──────────────────────────────────────────── */
.main .block-container {{
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1280px !important;
}}

/* ─── Global text overrides ──────────────────────────────────── */
p, span, div, li, td, th {{
    color: #1e293b;
}}

/* ─── White Card ─────────────────────────────────────────────── */
.lg-card {{
    background: #ffffff;
    border: 1px solid #dde8f8;
    border-radius: 20px;
    padding: 24px 28px;
    box-shadow:
        0 1px 3px rgba(29, 78, 216, 0.06),
        0 8px 32px rgba(29, 78, 216, 0.08);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
    position: relative;
    overflow: hidden;
}}

.lg-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {cfg["accent"]}, {cfg["accent2"]});
    border-radius: 20px 20px 0 0;
    pointer-events: none;
}}

.lg-card:hover {{
    border-color: {cfg["accent"]}55;
    box-shadow:
        0 2px 6px rgba(29, 78, 216, 0.08),
        0 12px 40px rgba(29, 78, 216, 0.12);
}}

/* ─── Hero Title ────────────────────────────────────────────── */
.hero-title {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    font-size: clamp(32px, 4vw, 52px);
    line-height: 1.05;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #1e293b 0%, {cfg["accent"]} 55%, {cfg["accent2"]} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
    transition: background 0.6s ease;
}}

.hero-sub {{
    font-size: 15px;
    color: #64748b;
    letter-spacing: 0.01em;
    font-weight: 400;
}}

/* ─── Mode Badge ────────────────────────────────────────────── */
.mode-badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 14px;
    border-radius: 999px;
    background: {cfg["tag_color"]};
    border: 1px solid {cfg["accent"]}33;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.8px;
    color: {cfg["tag_text"]};
    text-transform: uppercase;
    margin-bottom: 28px;
    margin-top: 8px;
}}

.mode-badge::before {{
    content: '';
    width: 7px; height: 7px;
    background: {cfg["accent"]};
    border-radius: 50%;
    box-shadow: 0 0 6px {cfg["accent"]}88;
    animation: pulse-dot 2s infinite;
}}

@keyframes pulse-dot {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(0.75); }}
}}

/* ─── Section Labels ────────────────────────────────────────── */
.section-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {cfg["accent"]};
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, {cfg["accent"]}44, transparent);
}}

/* ─── Upload Zone ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    border: 1.5px dashed {cfg["accent"]}55 !important;
    border-radius: 16px !important;
    background: {cfg["tag_color"]} !important;
    transition: border-color 0.3s, background 0.3s;
    padding: 8px !important;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: {cfg["accent"]}99 !important;
    background: {cfg["glow_strong"]} !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] {{
    color: #64748b !important;
    font-size: 14px !important;
}}

/* ─── Chat Input ────────────────────────────────────────────── */
[data-testid="stChatInput"] {{
    border: 1.5px solid {cfg["accent"]}44 !important;
    border-radius: 16px !important;
    background: #ffffff !important;
    box-shadow: 0 2px 12px {cfg["glow"]} !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}}

[data-testid="stChatInput"]:focus-within {{
    border-color: {cfg["accent"]}99 !important;
    box-shadow: 0 2px 20px {cfg["glow_strong"]} !important;
}}

[data-testid="stChatInput"] textarea {{
    color: #1e293b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    background: transparent !important;
}}

/* ─── Answer Box ────────────────────────────────────────────── */
.answer-box {{
    background: {cfg["tag_color"]};
    border-left: 3px solid {cfg["accent"]};
    border-radius: 0 12px 12px 0;
    padding: 20px 22px;
    font-size: 15px;
    line-height: 1.75;
    color: #1e293b;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
}}

/* ─── Confidence Bar ────────────────────────────────────────── */
.conf-row {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-top: 16px;
}}

.conf-bar-track {{
    flex: 1;
    height: 6px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
}}

.conf-bar-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, {cfg["accent"]}, {cfg["accent2"]});
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}}

.conf-pct {{
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 22px;
    color: {cfg["accent"]};
    min-width: 60px;
    text-align: right;
}}

/* ─── Evidence Cards ────────────────────────────────────────── */
.evidence-card {{
    background: #f8faff;
    border: 1px solid #dde8f8;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
}}

.evidence-card:hover {{
    background: #eef4ff;
    border-color: {cfg["accent"]}55;
}}

.evidence-meta {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
}}

.evidence-page {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {cfg["tag_text"]};
    background: {cfg["tag_color"]};
    border: 1px solid {cfg["accent"]}44;
    padding: 2px 8px;
    border-radius: 999px;
}}

.evidence-score {{
    font-size: 11px;
    color: #64748b;
    font-weight: 500;
}}

.evidence-text {{
    font-size: 13px;
    line-height: 1.65;
    color: #374151;
}}

/* ─── Chat Bubbles ──────────────────────────────────────────── */
.chat-wrap {{
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 24px;
}}

.bubble-user {{
    align-self: flex-end;
    max-width: 75%;
    background: {cfg["tag_color"]};
    border: 1px solid {cfg["accent"]}44;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    font-size: 14px;
    color: #1e293b;
}}

.bubble-bot {{
    align-self: flex-start;
    max-width: 85%;
    background: #ffffff;
    border: 1px solid #dde8f8;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    font-size: 14px;
    color: #374151;
}}

/* ─── Streamlit native text ──────────────────────────────────── */
.stMarkdown, .stMarkdown p, .stText, .stWrite {{
    color: #1e293b !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: #1e293b !important;
}}

/* ─── Streamlit caption & small text ────────────────────────── */
.stCaption, small, [data-testid="stCaptionContainer"] {{
    color: #64748b !important;
}}

/* ─── Buttons (native Streamlit) ─────────────────────────────── */
.stButton > button {{
    background: #ffffff !important;
    border: 1px solid #dde8f8 !important;
    color: #374151 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: background 0.2s, border-color 0.2s !important;
}}

.stButton > button:hover {{
    background: {cfg["tag_color"]} !important;
    border-color: {cfg["accent"]}55 !important;
    color: {cfg["accent"]} !important;
}}

/* ─── Download Button ───────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {{
    background: {cfg["tag_color"]} !important;
    border: 1px solid {cfg["accent"]}55 !important;
    color: {cfg["tag_text"]} !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px;
    padding: 8px 20px !important;
    transition: background 0.2s, box-shadow 0.2s !important;
}}

[data-testid="stDownloadButton"] > button:hover {{
    background: {cfg["accent"]} !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px {cfg["glow_strong"]} !important;
}}

/* ─── Spinner ───────────────────────────────────────────────── */
.stSpinner > div {{
    border-top-color: {cfg["accent"]} !important;
}}

/* ─── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: #f1f5f9; }}
::-webkit-scrollbar-thumb {{
    background: {cfg["accent"]}55;
    border-radius: 4px;
}}

/* ─── Success / Warning ─────────────────────────────────────── */
[data-testid="stSuccess"] {{
    background: #f0fdf4 !important;
    border-color: #86efac !important;
    border-radius: 12px !important;
    color: #166534 !important;
}}

[data-testid="stWarning"] {{
    background: #fffbeb !important;
    border-color: #fcd34d !important;
    border-radius: 12px !important;
    color: #92400e !important;
}}

/* ─── Progress bar override ─────────────────────────────────── */
[data-testid="stProgress"] > div > div > div {{
    background: linear-gradient(90deg, {cfg["accent"]}, {cfg["accent2"]}) !important;
}}

/* ─── Divider ───────────────────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    margin: 20px 0 !important;
}}

/* ─── Expander ──────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid #dde8f8 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}}

/* ─── Select / Radio in sidebar ─────────────────────────────── */
[data-testid="stSidebar"] [data-baseweb="select"] {{
    background: #f8faff !important;
}}

</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR CONTENT (after CSS so styles apply)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="
        margin-top: 16px;
        padding: 14px 16px;
        background: {cfg['tag_color']};
        border: 1px solid {cfg['accent']}33;
        border-radius: 14px;
        font-size: 13px;
        color: #374151;
        line-height: 1.6;
        transition: all 0.4s ease;
    ">
        <span style="color:{cfg['accent']}; font-weight:600;">
            top_k = {cfg['top_k']}
        </span><br>
        {cfg['desc']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:14px 16px; background:#f8faff;
         border:1px solid #dde8f8; border-radius:14px;">
        <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
             color:#94a3b8;margin-bottom:10px;font-weight:700;">How it works</div>
        <div style="font-size:12px;color:#4b5563;line-height:1.8;">
            ① Upload a research PDF<br>
            ② Choose your search mode<br>
            ③ Ask any question<br>
            ④ Get evidence-backed answers
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-title">Research Assistant</div>
<div class="hero-sub">Semantic search · Hybrid retrieval · Cross-encoder reranking</div>
<div class="mode-badge">{cfg['label']}&nbsp;MODE</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  UPLOAD SECTION
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📄 Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your research paper here",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file:
    if "pipeline_ready" not in st.session_state or \
       st.session_state.get("loaded_file") != uploaded_file.name:
        with st.spinner("⚙️ Parsing & indexing paper…"):
            docs = load_pdf(uploaded_file)
            chunks, index, bm25 = build_pipeline(docs)
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.bm25 = bm25
            st.session_state.pipeline_ready = True
            st.session_state.loaded_file = uploaded_file.name
        st.success(f"✅ **{uploaded_file.name}** indexed — {len(st.session_state.chunks)} semantic chunks ready.")
else:
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 4px; color: #94a3b8; font-size:13px;">
        No paper uploaded yet · Upload a PDF to begin
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ═══════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    st.markdown("### 🕘 Previous Questions")

    for i, (q, a) in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i

        if st.button(f"🆀 {q}", key=f"chat_{idx}"):

            # store selected chat
            st.session_state.selected_chat = idx
            st.session_state.show_dialog = True



# ═══════════════════════════════════════════════════════════════
#  DIALOG POPUP FOR PREVIOUS ANSWERS
# ═══════════════════════════════════════════════════════════════
if "show_dialog" not in st.session_state:
    st.session_state.show_dialog = False

if st.session_state.show_dialog and st.session_state.get("selected_chat") is not None:

    q, a = st.session_state.history[st.session_state.selected_chat]

    @st.dialog("📜 Previous Answer")
    def show_previous():
        st.markdown("### 🆀 Question")
        st.write(q)

        st.markdown("###  Answer")
        st.write(a)

        if st.button("Close"):
            st.session_state.show_dialog = False
            st.rerun()

    show_previous()

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ═══════════════════════════════════════════════════════════════
#  QUERY PROCESSING + RESULTS
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">🔍 Ask</div>', unsafe_allow_html=True)

# init states
if "query" not in st.session_state:
    st.session_state.query = None

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ── PROCESS QUERY ──────────────────────────────────────────────
if st.session_state.query:
    query = st.session_state.query

    if "pipeline_ready" not in st.session_state:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        with st.spinner("🔬 Retrieving, reranking, and synthesising…"):
            answer, results = generate_answer(
                query,
                st.session_state.chunks,
                st.session_state.index,
                st.session_state.bm25,
                top_k=top_k
            )
            top_sentences = extract_top_sentences(results, query)
            confidence = compute_confidence(top_sentences)

            # store history (limit to 5)
            st.session_state.history.append((query, answer))
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[-5:]

            # ✅ store CURRENT result (THIS FIXES YOUR BUG)
            st.session_state.current_result = {
                "query": query,
                "answer": answer,
                "top_sentences": top_sentences,
                "confidence": confidence
            }

    # clear trigger only (NOT the result)
    st.session_state.query = None


# ── DISPLAY RESULT (PERSISTENT) ────────────────────────────────
if st.session_state.current_result:

    data = st.session_state.current_result
    query = data["query"]
    answer = data["answer"]
    top_sentences = data["top_sentences"]
    confidence = data["confidence"]

    col_ans, col_ev = st.columns([11, 7], gap="large")

    # LEFT — Answer + Confidence
    with col_ans:
        st.markdown(f"""
        <div class="lg-card">
            <div class="section-label"> Answer</div>
            <div class="answer-box"><b> {query}</b><br><br>{answer}</div>
            <div class="section-label" style="margin-top:22px;">📊 Confidence</div>
            <div class="conf-row">
                <div class="conf-bar-track">
                    <div class="conf-bar-fill" style="width:{confidence}%;"></div>
                </div>
                <div class="conf-pct">{confidence}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT — Evidence
    with col_ev:
        st.markdown("### 🔎 Evidence")

        for score, sent, page in top_sentences[:5]:
            with st.container():
                c1, c2 = st.columns([1, 1])

                with c1:
                    st.caption(f"📄 Page {page}")

                with c2:
                    st.caption(f"⭐ Score: {round(score, 2)}")

                st.write(sent)
                st.divider()
        # ── Download ───────────────────────────────────────────
        st.download_button(
            "⬇ Download Result as JSON",
            json.dumps(
                {"query": query, "answer": answer, "confidence": confidence},
                indent=2
            ),
            file_name="research_result.json",
            mime="application/json"
        )

        # clear query after processing
        st.session_state.query = None


# ═══════════════════════════════════════════════════════════════
#  CHAT INPUT (MUST BE LAST)
# ═══════════════════════════════════════════════════════════════


user_input = st.chat_input("Ask anything about your paper…")

if user_input:
    st.session_state.query = user_input
    st.rerun()