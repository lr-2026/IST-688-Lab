import streamlit as st
import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # pysqlite3 not installed locally; system sqlite3 works

import csv
import json
import math
import tempfile
import time
from pathlib import Path

import chromadb
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder

st.title("Lab 8: Claim Checker (RAG + Re-ranking)")


# Settings

DATA_DIR = Path("./Lab-07-Data")           # put the SciFact files here
# NOTE: "./ChromaDB_for_Lab" resolves under /mount/src on Streamlit Cloud --
# that's the git-synced source checkout, not a safe place for a live SQLite-
# backed store. /tmp is guaranteed writable on Cloud; it just means the index
# rebuilds after a container restart instead of silently corrupting.
CHROMA_PATH = str(Path(tempfile.gettempdir()) / "ChromaDB_for_Lab")
COLLECTION_NAME = "SciFactChunks"          # new collection name for Lab 8
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
N_CANDIDATES = 30                          # stage 1 asks for this many chunks
TOP_N = 3                                  # abstracts that reach the prompt

# ---------------------------------------------------------------------------
# OpenAI client (kept from Lab 4)
# ---------------------------------------------------------------------------
if "client" not in st.session_state:
    api_key = (
        st.secrets.get("OPENAI_API_KEY")
        or st.secrets.get("my_key")
        or st.secrets.get("My_newkey")
    )
    if not api_key:
        st.error("No OpenAI key found in secrets (tried OPENAI_API_KEY, my_key, My_newkey).")
        st.stop()
    st.session_state.client = OpenAI(api_key=api_key)
client = st.session_state.client

openai_model = st.sidebar.selectbox("Which Model?", ("mini", "regular"))
model_to_use = "gpt-4o-mini" if openai_model == "mini" else "gpt-4o"

n_candidates_ui = st.sidebar.select_slider(
    "Stage 1 candidates (N_CANDIDATES)",
    options=[5, 10, 30, 50, 100],
    value=N_CANDIDATES,
    help=(
        "How many chunks stage 1 retrieves before re-ranking. The deck's own "
        "numbers (slide 17): recall keeps climbing past 30, but added latency "
        "goes from 48ms (5) to 1006ms (100) -- 30 is the lab's chosen knee of "
        "the curve, not a hard ceiling."
    ),
)


# ---------------------------------------------------------------------------
# Load SciFact data (replaces the PDF loading from Lab 4)
# Assumes the BEIR layout: corpus.jsonl, queries.jsonl, qrels/test.tsv
# ---------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data(show_spinner="Loading SciFact data...")
def load_scifact():
    corpus_path = next(DATA_DIR.rglob("corpus.jsonl"), None)
    queries_path = next(DATA_DIR.rglob("queries.jsonl"), None)
    qrels_path = next(DATA_DIR.rglob("test.tsv"), None)
    if not (corpus_path and queries_path and qrels_path):
        found = sorted(str(p) for p in DATA_DIR.rglob("*") if p.is_file())
        raise FileNotFoundError(
            "Expected corpus.jsonl, queries.jsonl and qrels/test.tsv under "
            f"{DATA_DIR}. Found: {found}"
        )

    abstracts = {}
    for row in read_jsonl(corpus_path):
        doc_id = str(row["_id"])
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        abstracts[doc_id] = f"{title}. {text}".strip()

    all_queries = {str(r["_id"]): r["text"] for r in read_jsonl(queries_path)}

    # Answer key: claim id -> set of relevant abstract ids
    qrels = {}
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 3 and int(float(row[2])) > 0:
                qrels.setdefault(str(row[0]), set()).add(str(row[1]))

    claims = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    return abstracts, claims, qrels


try:
    abstracts, claims, qrels = load_scifact()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# Chunking (new in Lab 7): 120 words, 30 shared with the next piece

def chunk_words(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = size - overlap
    return [
        " ".join(words[i:i + size])
        for i in range(0, max(len(words) - overlap, 1), step)
    ]


# Stage 0: build the ChromaDB collection (same idea as Lab 4 Part A)
# Chroma embeds with its default model (all-MiniLM-L6-v2): no OpenAI calls.

def _rebuild_collection(chroma_client, ids, docs, metas):
    """Delete (if present) and recreate the collection, then bulk-add."""
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet -- fine
    coll = chroma_client.create_collection(name=COLLECTION_NAME)
    progress = st.progress(0.0, text="Embedding chunks (first run only)...")
    batch = 500
    for start in range(0, len(ids), batch):
        coll.add(
            ids=ids[start:start + batch],
            documents=docs[start:start + batch],
            metadatas=metas[start:start + batch],
        )
        progress.progress(min((start + batch) / len(ids), 1.0))
    progress.empty()
    return coll


def build_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    ids, docs, metas = [], [], []
    for doc_id, text in abstracts.items():
        for i, chunk in enumerate(chunk_words(text)):
            ids.append(f"{doc_id}_{i}")
            docs.append(chunk)
            metas.append({"doc_id": doc_id})
    coll = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    if coll.count() != len(ids):
        # Empty, or incomplete because an earlier first run was interrupted:
        # start clean so the index always holds every chunk.
        try:
            coll = _rebuild_collection(chroma_client, ids, docs, metas)
        except chromadb.errors.NotFoundError:
            # Seen on Streamlit Cloud: the collection we just created isn't
            # immediately visible on the first attempt. One retry against a
            # fresh client has been enough in practice.
            chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            coll = _rebuild_collection(chroma_client, ids, docs, metas)
    return coll


if "SciFact_VectorDB" not in st.session_state:
    st.session_state.SciFact_VectorDB = build_collection()
collection = st.session_state.SciFact_VectorDB
st.sidebar.caption(
    f"Indexed {collection.count():,} chunks from {len(abstracts):,} abstracts "
    "(deck: 12,477 chunks from 5,183)"
)


# Stage 1: vector search, asking for MORE results (the one change vs Lab 4)

def stage1(question, n_results=N_CANDIDATES):
    results = collection.query(query_texts=[question], n_results=n_results)
    docs = results["documents"][0]
    doc_ids = [m["doc_id"] for m in results["metadatas"][0]]
    return doc_ids, docs


# Stage 2: re-rank (NEW). Cross-encoder reads question + chunk together.

@st.cache_resource(show_spinner="Loading re-ranker...")
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(question, doc_ids, docs):
    """Return (doc_ids, docs) reordered best-first."""
    pairs = [(question, d) for d in docs]
    scores = load_reranker().predict(pairs)
    order = np.argsort(-np.asarray(scores), kind="stable")
    return [doc_ids[i] for i in order], [docs[i] for i in order]


def unique_abstracts(doc_ids):
    """Chunks -> abstract ranking: keep first appearance of each abstract."""
    seen, out = set(), []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_chunk_position(question, target_id, search_pool=300):
    """
    Diagnostic only (not used in the verdict path): query stage 1 with a much
    wider pool than N_CANDIDATES and return the CHUNK position (1-based) of the
    first chunk from the answer-key abstract. N_CANDIDATES counts chunks, not
    abstracts, so this is the number it must reach to pull the abstract in.
    Returns None if the abstract is not in the top `search_pool` chunks.
    """
    doc_ids, _ = stage1(question, n_results=search_pool)
    if target_id in doc_ids:
        return doc_ids.index(target_id) + 1
    return None


def get_ranking(question, use_rerank, n_candidates=None):
    """Full abstract ranking for a claim, plus stage timings in ms."""
    t0 = time.perf_counter()
    doc_ids, docs = stage1(question, n_results=n_candidates or N_CANDIDATES)
    t1 = time.perf_counter()
    if use_rerank:
        doc_ids, docs = rerank(question, doc_ids, docs)
    t2 = time.perf_counter()
    return unique_abstracts(doc_ids), (t1 - t0) * 1000, (t2 - t1) * 1000

# Part C: evaluation metrics

def ndcg_at_k(ranked_ids, relevant, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranked_ids, relevant, k=3):
    return len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else 0.0


def evaluate(claim_ids, n_candidates=N_CANDIDATES):
    """Compare search-only vs re-ranked: retrieval quality and time."""
    out = {}
    for label, use_rerank in (("Search only", False), ("Re-ranked", True)):
        ndcgs, recalls, total_ms = [], [], []
        for cid in claim_ids:
            ranked, s1_ms, rr_ms = get_ranking(claims[cid], use_rerank, n_candidates)
            ndcgs.append(ndcg_at_k(ranked, qrels[cid], 10))
            recalls.append(recall_at_k(ranked, qrels[cid], TOP_N))
            total_ms.append(s1_ms + rr_ms)
        out[label] = {
            "nDCG@10": float(np.mean(ndcgs)),
            "Recall@3": float(np.mean(recalls)),
            "Median retrieval time (ms)": float(np.median(total_ms)),
        }
    out["Re-ranked"]["Added by re-ranking (ms)"] = (
        out["Re-ranked"]["Median retrieval time (ms)"]
        - out["Search only"]["Median retrieval time (ms)"]
    )
    out["Search only"]["Added by re-ranking (ms)"] = 0.0
    return out


# ---------------------------------------------------------------------------
# Part D: verdict prompt + generation
# ---------------------------------------------------------------------------
VERDICT_PROMPT = (
    "You check biomedical claims against research abstracts.\n"
    "Use only the abstracts below, never outside knowledge.\n"
    "Reply in exactly this format:\n"
    "Verdict: SUPPORTED, REFUTED, or NOT ENOUGH EVIDENCE\n"
    "Evidence: one or two sentences that cite abstracts by [id].\n"
    "An abstract can be on the same topic and still not address the claim. "
    "If none of them addresses it, the verdict is NOT ENOUGH EVIDENCE.\n\n"
    "--- ABSTRACTS ---\n{context}"
)


def get_verdict(claim, ranked_ids):
    context = "\n\n".join(f"[{d}] {abstracts[d]}" for d in ranked_ids[:TOP_N])
    resp = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": VERDICT_PROMPT.format(context=context)},
            {"role": "user", "content": f"Claim: {claim}"},
        ],
    )
    return resp.choices[0].message.content


def show_ids(ranked_ids, relevant):
    for i, d in enumerate(ranked_ids[:TOP_N], 1):
        tag = ""
        if relevant is not None:
            tag = "  (relevant)" if d in relevant else "  (not relevant)"
        st.write(f"{i}. [{d}] {abstracts[d][:90]}...{tag}")


tab_check, tab_eval = st.tabs(["Check a claim (A, B, D)", "Evaluate (C)"])

with tab_check:
    claim_ids = list(claims.keys())
    choice = st.selectbox(
        "Pick a SciFact test claim (answer key available) or write your own below",
        ["(write my own)"] + claim_ids,
        format_func=lambda c: c if c == "(write my own)" else f"{c}: {claims[c][:80]}",
    )
    default_text = "" if choice == "(write my own)" else claims[choice]
    claim_text = st.text_area("Claim", value=default_text, key=f"claim_{choice}")
    relevant = qrels.get(choice) if choice != "(write my own)" else None

    if st.button("Check claim") and claim_text.strip():
        with st.spinner("Searching, re-ranking, and generating..."):
            before_ids, s1_ms, _ = get_ranking(
                claim_text, use_rerank=False, n_candidates=n_candidates_ui
            )
            after_ids, _, rr_ms = get_ranking(
                claim_text, use_rerank=True, n_candidates=n_candidates_ui
            )
            before_verdict = get_verdict(claim_text, before_ids)
            after_verdict = get_verdict(claim_text, after_ids)

        st.caption(
            f"Stage 1 (N={n_candidates_ui}): {s1_ms:.0f} ms. "
            f"Re-ranking added: {rr_ms:.0f} ms."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Without re-ranking")
            show_ids(before_ids, relevant)
            st.info(before_verdict)
        with col2:
            st.subheader("With re-ranking")
            show_ids(after_ids, relevant)
            st.success(after_verdict)

        if relevant is not None:
            st.write("Answer key (relevant abstract IDs):", ", ".join(sorted(relevant)))
            st.write(
                f"Distinct abstracts among the {n_candidates_ui} candidate chunks: "
                f"{len(before_ids)}"
            )
            for r in sorted(relevant):
                rank_before = before_ids.index(r) + 1 if r in before_ids else None
                rank_after = after_ids.index(r) + 1 if r in after_ids else None
                if rank_before is None:
                    st.warning(
                        f"[{r}] was never found among the top {n_candidates_ui} "
                        "candidate chunks, so re-ranking cannot help "
                        "(it can only reorder what stage 1 retrieved)."
                    )
                    with st.spinner("Checking a wider pool for diagnosis..."):
                        pos = find_chunk_position(claim_text, r, search_pool=300)
                    if pos is None:
                        st.caption(
                            f"[{r}] does not appear even in the top 300 chunks. "
                            "This looks like a genuine similarity gap "
                            "(surface vocabulary mismatch), not just a width problem."
                        )
                    else:
                        st.caption(
                            f"[{r}] first appears at chunk position {pos} in a wider "
                            f"search. Raising N_CANDIDATES to {pos} or above would "
                            "bring it into the candidates; re-ranking alone cannot."
                        )
                else:
                    st.write(
                        f"[{r}] abstract rank before re-ranking: {rank_before}, "
                        f"after: {rank_after}"
                    )

with tab_eval:
    st.write(
        "Scores retrieval before and after re-ranking against the SciFact "
        f"answer key, using N_CANDIDATES = {n_candidates_ui} from the sidebar. "
        "Using fewer claims is faster."
    )
    n_claims = st.slider("Number of test claims", 10, len(claims), min(50, len(claims)), step=10)
    if st.button("Run evaluation"):
        subset = list(claims.keys())[:n_claims]
        with st.spinner(f"Evaluating {n_claims} claims..."):
            results = evaluate(subset, n_candidates=n_candidates_ui)
        st.table(results)
        st.caption(
            "Deck reference (300 claims, N=30, laptop): nDCG@10 0.672 -> 0.734, "
            "Recall@3 0.683 -> 0.724, about +318 ms."
        )
