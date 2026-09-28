"""
g1.py — [G1] Generación y almacenamiento de embeddings con Pgvector.

Mismo flujo que p1.py, pero el embedding se guarda como `vector(384)`
(tipo nativo de la extensión), no como REAL[].

Mide por separado:
  - generación (modelo)
  - almacenamiento (UPDATE)  ← métrica del enunciado
Tras cargar, crea índices HNSW (cosine + L2) para las consultas nativas de g2.
"""

import statistics
import sys
import time
from typing import List, Tuple

import psycopg2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE = "corpus_pgvector"
EMBEDDING_DIM = 384


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def embed_phrase(tokenizer, model, phrase: str) -> List[float]:
    """Mismo pipeline que p1/c1 → resultados comparables."""
    encoded = tokenizer(
        phrase,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    with torch.no_grad():
        model_output = model(**encoded)
        embedding = mean_pooling(model_output, encoded["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding[0].cpu().tolist()


def to_vector_literal(embedding: List[float]) -> str:
    """Literal Pgvector: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def print_time_metrics(times: List[float], label: str) -> None:
    if not times:
        print(f"[AVISO] No hay mediciones para '{label}'.")
        return
    std_val = statistics.pstdev(times) if len(times) == 1 else statistics.stdev(times)
    print("\n" + "=" * 60)
    print(f"Resultados [G1]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def fetch_phrases(conn) -> List[Tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, frase FROM {TABLE} ORDER BY id;")
        return cur.fetchall()


def store_embedding(conn, row_id: int, embedding: List[float]) -> None:
    """UPDATE con cast a vector(N) — sin sacar datos a Python en g2."""
    lit = to_vector_literal(embedding)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {TABLE} SET embedding = %s::vector WHERE id = %s;",
            (lit, row_id),
        )
    conn.commit()


def create_indexes(conn) -> None:
    """
    Índices HNSW nativos de Pgvector (uno por métrica de distancia).
    No se incluyen en las métricas de almacenamiento del enunciado.
    """
    with conn.cursor() as cur:
        print("      Creando índice HNSW cosine (vector_cosine_ops)...")
        t0 = time.perf_counter()
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {TABLE}_embedding_cosine_idx
            ON {TABLE}
            USING hnsw (embedding vector_cosine_ops);
            """
        )
        conn.commit()
        print(f"      Índice cosine OK ({time.perf_counter() - t0:.3f} s)")

        print("      Creando índice HNSW L2 (vector_l2_ops)...")
        t0 = time.perf_counter()
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {TABLE}_embedding_l2_idx
            ON {TABLE}
            USING hnsw (embedding vector_l2_ops);
            """
        )
        conn.commit()
        print(f"      Índice L2 OK ({time.perf_counter() - t0:.3f} s)")


def main() -> int:
    conn = None
    try:
        print("[1/4] Conectando a PostgreSQL (Pgvector)...")
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_extension WHERE extname = %s;",
                ("vector",),
            )
            if cur.fetchone() is None:
                print(
                    "[ERROR] Extensión 'vector' no activa. Ejecuta primero g0.py "
                    "(y asegúrate de tener Pgvector instalado en el servidor).",
                    file=sys.stderr,
                )
                return 1
            cur.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name = %s;
                """,
                (TABLE,),
            )
            if cur.fetchone() is None:
                print(
                    f"[ERROR] No existe la tabla '{TABLE}'. Ejecuta g0.py primero.",
                    file=sys.stderr,
                )
                return 1
        print("      Conexión OK.")

        print(f"[2/4] Cargando modelo '{MODEL_NAME}'...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        print("      Modelo cargado.")

        print(f"[3/4] Recuperando frases de {TABLE}...")
        rows = fetch_phrases(conn)
        if not rows:
            print("[ERROR] Tabla vacía. Ejecuta g0.py primero.", file=sys.stderr)
            return 1
        print(f"      {len(rows)} frases.")

        print("[4/4] Generando embeddings y almacenándolos como vector(384)...")
        times_gen: List[float] = []
        times_store: List[float] = []

        for i, (row_id, frase) in enumerate(rows, start=1):
            t_gen0 = time.perf_counter()
            embedding = embed_phrase(tokenizer, model, frase)
            if len(embedding) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Dimensión inesperada {len(embedding)} (esperada {EMBEDDING_DIM})."
                )
            times_gen.append(time.perf_counter() - t_gen0)

            t_store0 = time.perf_counter()
            store_embedding(conn, row_id, embedding)
            times_store.append(time.perf_counter() - t_store0)

            if i % 500 == 0 or i == len(rows):
                print(
                    f"      Progreso: {i}/{len(rows)} | "
                    f"gen={times_gen[-1]:.4f}s store={times_store[-1]:.4f}s"
                )

        print_time_metrics(times_gen, "generación de embeddings (modelo)")
        print_time_metrics(
            times_store,
            "almacenamiento de embeddings / UPDATE vector (storing embeddings)",
        )
        times_total = [g + s for g, s in zip(times_gen, times_store)]
        print_time_metrics(times_total, "ciclo completo (generación + almacenamiento)")

        print("\nCreando índices HNSW para g2 (fuera de las métricas de store)...")
        create_indexes(conn)

        print("\n[G1] Completado. Siguiente paso: g2.py")
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
