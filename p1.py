"""
p1.py — [P1] Generación y almacenamiento de embeddings en PostgreSQL.

Enunciado: conectar a la BD, generar embeddings por frase y almacenarlos.
Medir min / max / avg / std del tiempo de almacenamiento de embeddings.
Sin Pgvector: se usa REAL[] (impedance mismatch: vector → array SQL).

Se miden por separado:
  - generación del embedding (modelo, en Python)
  - almacenamiento (UPDATE en PostgreSQL)  ← métrica principal del enunciado
"""

import statistics
import sys
import time
from typing import List, Tuple

import psycopg2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuración (misma BD que p0.py)
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling ponderado por attention mask (recomendado por Sentence-Transformers)."""
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def embed_phrase(tokenizer, model, phrase: str) -> List[float]:
    """Embedding L2-normalizado de una frase (lista de floats para REAL[])."""
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


def print_time_metrics(times: List[float], label: str) -> None:
    """Imprime min, max, avg y std según el enunciado del laboratorio."""
    if not times:
        print(f"[AVISO] No hay mediciones para '{label}'.")
        return

    std_val = statistics.pstdev(times) if len(times) == 1 else statistics.stdev(times)
    print("\n" + "=" * 60)
    print(f"Resultados [P1]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def fetch_phrases(conn) -> List[Tuple[int, str]]:
    """Recupera (id, frase) ya insertados por p0.py."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, frase FROM corpus ORDER BY id;")
        return cur.fetchall()


def store_embedding(conn, row_id: int, embedding: List[float]) -> None:
    """Persiste el embedding como REAL[] (sin Pgvector)."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE corpus SET embedding = %s WHERE id = %s;",
            (embedding, row_id),
        )
    conn.commit()


def main() -> int:
    conn = None
    try:
        print("[1/4] Conectando a PostgreSQL...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("      Conexión OK.")

        print(f"[2/4] Cargando modelo '{MODEL_NAME}'...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        print("      Modelo cargado.")

        print("[3/4] Recuperando frases de corpus...")
        rows = fetch_phrases(conn)
        if not rows:
            print("[ERROR] Tabla corpus vacía. Ejecuta primero p0.py.", file=sys.stderr)
            return 1
        print(f"      {len(rows)} frases encontradas.")

        print("[4/4] Generando embeddings y almacenándolos en PostgreSQL...")
        times_gen: List[float] = []
        times_store: List[float] = []

        for i, (row_id, frase) in enumerate(rows, start=1):
            # --- Generación (fuera de la BD; coste del modelo) ---
            t_gen0 = time.perf_counter()
            embedding = embed_phrase(tokenizer, model, frase)
            times_gen.append(time.perf_counter() - t_gen0)

            # --- Almacenamiento (métrica principal del enunciado) ---
            t_store0 = time.perf_counter()
            store_embedding(conn, row_id, embedding)
            times_store.append(time.perf_counter() - t_store0)

            if i % 500 == 0 or i == len(rows):
                print(
                    f"      Progreso: {i}/{len(rows)} | "
                    f"gen={times_gen[-1]:.4f}s store={times_store[-1]:.4f}s"
                )

        # Separar gen vs store permite discutir impedance mismatch y
        # responder si el coste dominante es el modelo o el mapeo a REAL[].
        print_time_metrics(times_gen, "generación de embeddings (modelo)")
        print_time_metrics(
            times_store,
            "almacenamiento de embeddings / UPDATE REAL[] (storing embeddings)",
        )

        times_total = [g + s for g, s in zip(times_gen, times_store)]
        print_time_metrics(times_total, "ciclo completo (generación + almacenamiento)")

        print("\n[P1] Completado. Siguiente paso: p2.py")
        return 0

    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Inesperado: {e}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()
            print("Conexión cerrada.")


if __name__ == "__main__":
    sys.exit(main())
