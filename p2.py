"""
p2.py — [P2] Consultas de similitud en PostgreSQL (impedance mismatch).

Enunciado:
  - Elegir 10 frases claramente identificadas.
  - Para cada una, hallar las top-2 más similares entre el resto del corpus.
  - Usar DOS métricas: Distancia Euclidiana y Similitud del Coseno.
  - SIN Pgvector: se traen todos los embeddings a Python y se calculan
    las distancias fuera de la BD (esto fuerza el impedance mismatch).
  - Medir min / max / avg / std del tiempo de cómputo top-2.
"""

import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import psycopg2
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Configuración (misma BD que p0.py / p1.py)
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}

# 10 frases objetivo FIJAS y reproducibles (mismas IDs se reutilizarán en c2.py).
# Distribuidas a lo largo del corpus (~10k). Se validan al arrancar.
QUERY_IDS: List[int] = [1, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]


def print_time_metrics(times: List[float], label: str) -> None:
    """Imprime min, max, avg y std según el enunciado del laboratorio."""
    if not times:
        print(f"[AVISO] No hay mediciones para '{label}'.")
        return

    std_val = statistics.pstdev(times) if len(times) == 1 else statistics.stdev(times)
    print("\n" + "=" * 60)
    print(f"Resultados [P2]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def fetch_all_embeddings(
    conn,
) -> Tuple[List[int], List[str], np.ndarray]:
    """
    Extrae TODOS los (id, frase, embedding) a memoria Python.

    Impedance mismatch: PostgreSQL no calcula vecinos vectoriales (sin Pgvector),
    así que hay que materializar los REAL[] en el cliente y operar en NumPy/SciPy.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, frase, embedding
            FROM corpus
            WHERE embedding IS NOT NULL
            ORDER BY id;
            """
        )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            "No hay embeddings en corpus. Ejecuta p0.py y luego p1.py primero."
        )

    ids: List[int] = []
    frases: List[str] = []
    vectors: List[Sequence[float]] = []

    for row_id, frase, embedding in rows:
        if embedding is None:
            continue
        ids.append(int(row_id))
        frases.append(frase)
        vectors.append(list(embedding))

    matrix = np.asarray(vectors, dtype=np.float64)
    return ids, frases, matrix


def top2_euclidean(
    query_idx: int, matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-2 por Distancia Euclidiana (menor distancia = más similar).
    Excluye la propia frase consulta.
    Devuelve (índices locales top-2, distancias).
    """
    dists = cdist(matrix[query_idx : query_idx + 1], matrix, metric="euclidean")[0]
    dists[query_idx] = np.inf  # excluir self
    top_idx = np.argpartition(dists, 2)[:2]
    top_idx = top_idx[np.argsort(dists[top_idx])]
    return top_idx, dists[top_idx]


def top2_cosine_similarity(
    query_idx: int, matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-2 por Similitud del Coseno (mayor similitud = más similar).
    SciPy ofrece cosine *distance* (= 1 - similarity); convertimos a similitud.
    Excluye la propia frase consulta.
    Devuelve (índices locales top-2, similitudes).
    """
    cos_dist = cdist(matrix[query_idx : query_idx + 1], matrix, metric="cosine")[0]
    similarity = 1.0 - cos_dist
    similarity[query_idx] = -np.inf  # excluir self
    top_idx = np.argpartition(-similarity, 2)[:2]  # los 2 mayores
    top_idx = top_idx[np.argsort(-similarity[top_idx])]
    return top_idx, similarity[top_idx]


def resolve_query_indices(
    all_ids: List[int], query_ids: List[int]
) -> List[Tuple[int, int]]:
    """
    Mapea QUERY_IDS (id de BD) → índice local en la matriz.
    Falla si algún id no existe o no tiene embedding.
    """
    id_to_local: Dict[int, int] = {row_id: i for i, row_id in enumerate(all_ids)}
    resolved: List[Tuple[int, int]] = []
    missing: List[int] = []

    for qid in query_ids:
        if qid not in id_to_local:
            missing.append(qid)
        else:
            resolved.append((qid, id_to_local[qid]))

    if missing:
        raise RuntimeError(
            f"QUERY_IDS no encontrados (o sin embedding): {missing}. "
            f"IDs disponibles: {all_ids[0]}..{all_ids[-1]} (n={len(all_ids)})."
        )
    return resolved


def print_neighbors(
    metric_name: str,
    query_db_id: int,
    query_text: str,
    neighbor_ids: List[int],
    neighbor_texts: List[str],
    scores: np.ndarray,
    score_label: str,
) -> None:
    """Imprime de forma clara la consulta y sus top-2 (exigido por el enunciado)."""
    preview_q = query_text if len(query_text) <= 100 else query_text[:97] + "..."
    print(f"\n--- Query id={query_db_id} | métrica: {metric_name} ---")
    print(f"  Frase: {preview_q}")
    for rank, (nid, ntext, score) in enumerate(
        zip(neighbor_ids, neighbor_texts, scores), start=1
    ):
        preview_n = ntext if len(ntext) <= 100 else ntext[:97] + "..."
        print(f"  Top-{rank}: id={nid} | {score_label}={score:.6f}")
        print(f"           {preview_n}")


def main() -> int:
    conn = None
    try:
        print("[1/3] Conectando a PostgreSQL...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("      Conexión OK.")

        print("[2/3] Extrayendo TODOS los embeddings a Python (impedance mismatch)...")
        t_fetch0 = time.perf_counter()
        all_ids, all_frases, matrix = fetch_all_embeddings(conn)
        t_fetch = time.perf_counter() - t_fetch0
        print(
            f"      {len(all_ids)} vectores cargados | shape={matrix.shape} | "
            f"fetch={t_fetch:.4f} s"
        )

        queries = resolve_query_indices(all_ids, QUERY_IDS)
        print(f"[3/3] Consultas objetivo (QUERY_IDS): {QUERY_IDS}")

        times_euclidean: List[float] = []
        times_cosine: List[float] = []
        times_both: List[float] = []  # ambas métricas por frase (métrica global del lab)

        for qid, local_idx in queries:
            t_both0 = time.perf_counter()

            # --- Distancia Euclidiana ---
            t0 = time.perf_counter()
            idx_e, dist_e = top2_euclidean(local_idx, matrix)
            times_euclidean.append(time.perf_counter() - t0)

            print_neighbors(
                metric_name="Distancia Euclidiana",
                query_db_id=qid,
                query_text=all_frases[local_idx],
                neighbor_ids=[all_ids[i] for i in idx_e],
                neighbor_texts=[all_frases[i] for i in idx_e],
                scores=dist_e,
                score_label="euclidean_dist",
            )

            # --- Similitud del Coseno ---
            t0 = time.perf_counter()
            idx_c, sim_c = top2_cosine_similarity(local_idx, matrix)
            times_cosine.append(time.perf_counter() - t0)

            print_neighbors(
                metric_name="Similitud del Coseno",
                query_db_id=qid,
                query_text=all_frases[local_idx],
                neighbor_ids=[all_ids[i] for i in idx_c],
                neighbor_texts=[all_frases[i] for i in idx_c],
                scores=sim_c,
                score_label="cosine_sim",
            )

            times_both.append(time.perf_counter() - t_both0)

        # Métricas exigidas: top-2 para las 10 frases (y desglose por métrica → PQ1)
        print_time_metrics(
            times_both,
            "cómputo top-2 (Euclidiana + Coseno) por frase consulta",
        )
        print_time_metrics(
            times_euclidean,
            "cómputo top-2 — Distancia Euclidiana",
        )
        print_time_metrics(
            times_cosine,
            "cómputo top-2 — Similitud del Coseno",
        )
        print(
            f"\nTiempo único de extracción de embeddings (no incluido arriba): "
            f"{t_fetch:.6f} s"
        )
        print("\n[P2] Completado.")
        return 0

    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()
            print("Conexión cerrada.")


if __name__ == "__main__":
    sys.exit(main())
