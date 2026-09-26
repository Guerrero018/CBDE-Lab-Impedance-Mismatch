"""
c2.py — [C2] Consultas de similitud nativas en ChromaDB.

Enunciado:
  - Mismas 10 frases que p2.py (QUERY_IDS).
  - Top-2 más similares con DOS métricas: Euclidiana y Coseno.
  - Usar las capacidades nativas de Chroma (collection.query + índice HNSW).
  - Medir min / max / avg / std del tiempo de cómputo top-2.

Nota sobre métricas en Chroma:
  Cada colección fija UNA distancia en el índice (hnsw:space).
  - 'corpus'     → cosine  (creada en c0)
  - 'corpus_l2'  → l2      (réplica de embeddings para Distancia Euclidiana)
  Así ambas métricas se resuelven con query nativa, comparable a p2 pero sin
  traer los 10k vectores a Python en cada consulta.
"""

import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb

# ---------------------------------------------------------------------------
# Configuración (alineada con c0/c1 y con p2.py)
# ---------------------------------------------------------------------------
CHROMA_PATH = Path("chroma_db")
COLLECTION_COSINE = "corpus"
COLLECTION_L2 = "corpus_l2"
BATCH_SIZE = 500

# Mismas frases objetivo que p2.py (reproducibilidad cruzada PostgreSQL ↔ Chroma)
QUERY_IDS: List[int] = [1, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]


def print_time_metrics(times: List[float], label: str) -> None:
    """Imprime min, max, avg y std según el enunciado del laboratorio."""
    if not times:
        print(f"[AVISO] No hay mediciones para '{label}'.")
        return

    std_val = statistics.pstdev(times) if len(times) == 1 else statistics.stdev(times)
    print("\n" + "=" * 60)
    print(f"Resultados [C2]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def ensure_l2_collection(
    client: chromadb.PersistentClient,
    source: chromadb.Collection,
) -> chromadb.Collection:
    """
    Crea/recrea 'corpus_l2' con los mismos ids/docs/embeddings que 'corpus',
    pero con hnsw:space=l2 para consultas nativas de Distancia Euclidiana.
    (No forma parte de las métricas de query del enunciado.)
    """
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_L2 in existing:
        client.delete_collection(COLLECTION_L2)

    col_l2 = client.create_collection(
        name=COLLECTION_L2,
        embedding_function=None,
        metadata={"hnsw:space": "l2"},
    )

    data = source.get(include=["documents", "embeddings"])
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    embeddings = data.get("embeddings") or []

    if not ids:
        raise RuntimeError(
            f"Colección '{COLLECTION_COSINE}' vacía. Ejecuta c0.py y c1.py primero."
        )

    print(f"      Copiando {len(ids)} items a '{COLLECTION_L2}' (espacio l2)...")
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        col_l2.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
        )
        print(f"      L2 progreso: {min(end, len(ids))}/{len(ids)}")

    return col_l2


def get_query_item(
    collection: chromadb.Collection, query_id: str
) -> Tuple[str, List[float]]:
    """Obtiene (documento, embedding) de una frase consulta por id."""
    got = collection.get(ids=[query_id], include=["documents", "embeddings"])
    if not got["ids"]:
        raise RuntimeError(f"QUERY_ID {query_id} no existe en Chroma.")
    doc = got["documents"][0]
    emb = got["embeddings"][0]
    if emb is None:
        raise RuntimeError(
            f"QUERY_ID {query_id} no tiene embedding. Ejecuta c1.py primero."
        )
    return doc, list(emb)


def top2_native(
    collection: chromadb.Collection,
    query_id: str,
    query_embedding: List[float],
) -> List[Tuple[str, Optional[str], float]]:
    """
    Top-2 vecinos vía query nativa de Chroma.
    Pide 3 resultados porque el más cercano suele ser la propia consulta;
    luego se excluye self.
    Devuelve lista de (id, document, distance).
    """
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    neighbors: List[Tuple[str, Optional[str], float]] = []
    for nid, ndoc, dist in zip(ids, docs, dists):
        if nid == query_id:
            continue
        neighbors.append((nid, ndoc, float(dist)))
        if len(neighbors) == 2:
            break

    if len(neighbors) < 2:
        # Fallback por si self no estaba en el top-3 (poco habitual)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        neighbors = []
        for nid, ndoc, dist in zip(ids, docs, dists):
            if nid == query_id:
                continue
            neighbors.append((nid, ndoc, float(dist)))
            if len(neighbors) == 2:
                break

    return neighbors


def print_neighbors(
    metric_name: str,
    query_id: str,
    query_text: str,
    neighbors: List[Tuple[str, Optional[str], float]],
    score_label: str,
    score_transform=None,
) -> None:
    """Imprime consulta + top-2 (exigido por el enunciado)."""
    preview_q = query_text if len(query_text) <= 100 else query_text[:97] + "..."
    print(f"\n--- Query id={query_id} | métrica: {metric_name} ---")
    print(f"  Frase: {preview_q}")
    for rank, (nid, ntext, dist) in enumerate(neighbors, start=1):
        score = score_transform(dist) if score_transform else dist
        preview_n = (ntext or "")[:100]
        if ntext and len(ntext) > 100:
            preview_n = ntext[:97] + "..."
        print(f"  Top-{rank}: id={nid} | {score_label}={score:.6f}")
        print(f"           {preview_n}")


def main() -> int:
    try:
        print(f"[1/4] Abriendo PersistentClient en '{CHROMA_PATH}'...")
        if not CHROMA_PATH.exists():
            print(
                f"[ERROR] No existe '{CHROMA_PATH}'. Ejecuta c0.py y c1.py.",
                file=sys.stderr,
            )
            return 1

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        col_cosine = client.get_collection(
            name=COLLECTION_COSINE,
            embedding_function=None,
        )
        print(f"      '{COLLECTION_COSINE}' OK (n={col_cosine.count()}, space=cosine).")

        print("[2/4] Preparando colección L2 para Distancia Euclidiana nativa...")
        t_setup0 = time.perf_counter()
        col_l2 = ensure_l2_collection(client, col_cosine)
        t_setup = time.perf_counter() - t_setup0
        print(f"      '{COLLECTION_L2}' OK (n={col_l2.count()}) | setup={t_setup:.4f} s")

        print(f"[3/4] Consultas objetivo (QUERY_IDS): {QUERY_IDS}")
        print("[4/4] Ejecutando top-2 nativo por métrica...")

        times_euclidean: List[float] = []
        times_cosine: List[float] = []
        times_both: List[float] = []

        for qid_int in QUERY_IDS:
            qid = str(qid_int)
            query_text, query_emb = get_query_item(col_cosine, qid)

            t_both0 = time.perf_counter()

            # --- Distancia Euclidiana (colección l2, query nativa) ---
            t0 = time.perf_counter()
            neigh_e = top2_native(col_l2, qid, query_emb)
            times_euclidean.append(time.perf_counter() - t0)

            print_neighbors(
                metric_name="Distancia Euclidiana (Chroma nativo, space=l2)",
                query_id=qid,
                query_text=query_text or "",
                neighbors=neigh_e,
                score_label="l2_dist",
            )

            # --- Similitud del Coseno (colección cosine; Chroma devuelve distance) ---
            t0 = time.perf_counter()
            neigh_c = top2_native(col_cosine, qid, query_emb)
            times_cosine.append(time.perf_counter() - t0)

            print_neighbors(
                metric_name="Similitud del Coseno (Chroma nativo, space=cosine)",
                query_id=qid,
                query_text=query_text or "",
                neighbors=neigh_c,
                score_label="cosine_sim",
                # Chroma reporta cosine *distance*; convertimos a similitud como en p2
                score_transform=lambda d: 1.0 - d,
            )

            times_both.append(time.perf_counter() - t_both0)

        print_time_metrics(
            times_both,
            "cómputo top-2 nativo (Euclidiana + Coseno) por frase consulta",
        )
        print_time_metrics(
            times_euclidean,
            "cómputo top-2 nativo — Distancia Euclidiana (l2)",
        )
        print_time_metrics(
            times_cosine,
            "cómputo top-2 nativo — Similitud del Coseno",
        )
        print(
            f"\nTiempo de setup colección L2 (no incluido arriba): {t_setup:.6f} s"
        )
        print("\n[C2] Completado.")
        return 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
