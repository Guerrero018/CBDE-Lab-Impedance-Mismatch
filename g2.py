"""
g2.py — [G2] Consultas top-2 nativas con Pgvector.

Mismas 10 frases que p2/c2 (QUERY_IDS). Dos métricas:
  - Distancia Euclidiana: operador <->  (L2)
  - Similitud del Coseno: operador <=>  (cosine distance; sim = 1 - dist)

A diferencia de p2, el k-NN se calcula EN PostgreSQL (sin traer 10k vectores
a Python). Eso reduce el impedance mismatch respecto a REAL[] + SciPy.
"""

import statistics
import sys
import time
from typing import List, Optional, Tuple

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}
TABLE = "corpus_pgvector"

# Mismos IDs que p2.py / c2.py
QUERY_IDS: List[int] = [1, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]


def print_time_metrics(times: List[float], label: str) -> None:
    if not times:
        print(f"[AVISO] No hay mediciones para '{label}'.")
        return
    std_val = statistics.pstdev(times) if len(times) == 1 else statistics.stdev(times)
    print("\n" + "=" * 60)
    print(f"Resultados [G2]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def get_query_row(conn, query_id: int) -> Tuple[str, str]:
    """Devuelve (frase, literal vector) de la consulta."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT frase, embedding::text FROM {TABLE} WHERE id = %s;",
            (query_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"QUERY_ID {query_id} no existe en {TABLE}.")
    frase, emb_text = row
    if emb_text is None:
        raise RuntimeError(f"QUERY_ID {query_id} sin embedding. Ejecuta g1.py.")
    return frase, emb_text


def top2_euclidean(conn, query_id: int, query_vec: str) -> List[Tuple[int, str, float]]:
    """
    Top-2 por Distancia Euclidiana nativa (<->).
    ORDER BY ... <-> query usa el índice HNSW L2 si existe.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, frase, (embedding <-> %s::vector) AS dist
            FROM {TABLE}
            WHERE id <> %s AND embedding IS NOT NULL
            ORDER BY embedding <-> %s::vector
            LIMIT 2;
            """,
            (query_vec, query_id, query_vec),
        )
        return [(int(r[0]), r[1], float(r[2])) for r in cur.fetchall()]


def top2_cosine(conn, query_id: int, query_vec: str) -> List[Tuple[int, str, float]]:
    """
    Top-2 por distancia del coseno nativa (<=>).
    Devolvemos similitud = 1 - distance para alinear con p2/c2.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, frase, (embedding <=> %s::vector) AS dist
            FROM {TABLE}
            WHERE id <> %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 2;
            """,
            (query_vec, query_id, query_vec),
        )
        rows = cur.fetchall()
    return [(int(r[0]), r[1], 1.0 - float(r[2])) for r in rows]


def print_neighbors(
    metric_name: str,
    query_id: int,
    query_text: str,
    neighbors: List[Tuple[int, str, float]],
    score_label: str,
) -> None:
    preview_q = query_text if len(query_text) <= 100 else query_text[:97] + "..."
    print(f"\n--- Query id={query_id} | métrica: {metric_name} ---")
    print(f"  Frase: {preview_q}")
    for rank, (nid, ntext, score) in enumerate(neighbors, start=1):
        preview_n = ntext if len(ntext) <= 100 else ntext[:97] + "..."
        print(f"  Top-{rank}: id={nid} | {score_label}={score:.6f}")
        print(f"           {preview_n}")


def main() -> int:
    conn = None
    try:
        print("[1/2] Conectando a PostgreSQL (Pgvector)...")
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {TABLE} WHERE embedding IS NOT NULL;"
            )
            n = cur.fetchone()[0]
        if n == 0:
            print(
                "[ERROR] No hay embeddings. Ejecuta g0.py y g1.py primero.",
                file=sys.stderr,
            )
            return 1
        print(f"      OK — {n} embeddings en '{TABLE}'.")

        print(f"[2/2] Consultas objetivo (QUERY_IDS): {QUERY_IDS}")
        times_euclidean: List[float] = []
        times_cosine: List[float] = []
        times_both: List[float] = []

        for qid in QUERY_IDS:
            query_text, query_vec = get_query_row(conn, qid)
            t_both0 = time.perf_counter()

            t0 = time.perf_counter()
            neigh_e = top2_euclidean(conn, qid, query_vec)
            times_euclidean.append(time.perf_counter() - t0)
            print_neighbors(
                "Distancia Euclidiana (Pgvector <->)",
                qid,
                query_text,
                neigh_e,
                "euclidean_dist",
            )

            t0 = time.perf_counter()
            neigh_c = top2_cosine(conn, qid, query_vec)
            times_cosine.append(time.perf_counter() - t0)
            print_neighbors(
                "Similitud del Coseno (Pgvector <=>)",
                qid,
                query_text,
                neigh_c,
                "cosine_sim",
            )

            times_both.append(time.perf_counter() - t_both0)

        print_time_metrics(
            times_both,
            "cómputo top-2 nativo (Euclidiana + Coseno) por frase consulta",
        )
        print_time_metrics(
            times_euclidean,
            "cómputo top-2 nativo — Distancia Euclidiana (<->)",
        )
        print_time_metrics(
            times_cosine,
            "cómputo top-2 nativo — Similitud del Coseno (<=>)",
        )
        print("\n[G2] Completado.")
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
