"""
g0.py — [G0] Carga de texto en PostgreSQL + Pgvector.

Parte opcional del lab: mismos experimentos que p0/c0, pero con la extensión
nativa `vector` (https://github.com/pgvector/pgvector).

- Crea la extensión si falta (debe estar instalada en el servidor).
- Usa tabla propia `corpus_pgvector` (no pisa la tabla `corpus` de p0/p1).
- Inserta solo texto y mide min / max / avg / std.
"""

import statistics
import sys
import time
from pathlib import Path

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}
CORPUS_FILE = Path("bookcorpus_10k.txt")
# all-MiniLM-L6-v2 → 384 dimensiones
EMBEDDING_DIM = 384
TABLE = "corpus_pgvector"


def print_time_metrics(times: list[float], label: str) -> None:
    if not times:
        print(f"[AVISO] No hay mediciones para '{label}'.")
        return
    std_val = statistics.pstdev(times) if len(times) == 1 else statistics.stdev(times)
    print("\n" + "=" * 60)
    print(f"Resultados [G0]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def load_sentences(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró '{path}'. Colócalo en el directorio del laboratorio."
        )
    with path.open("r", encoding="utf-8") as f:
        frases = [line.strip() for line in f if line.strip()]
    if not frases:
        raise ValueError(f"El archivo '{path}' no contiene frases.")
    return frases


def ensure_pgvector_schema(conn) -> None:
    """Activa la extensión y (re)crea la tabla tipada con vector(N)."""
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except psycopg2.Error as e:
            conn.rollback()
            raise RuntimeError(
                "La extensión Pgvector no está disponible en este PostgreSQL.\n"
                "Instálala siguiendo https://github.com/pgvector/pgvector "
                "(Windows: compilar o usar el instalador de tu distribución) "
                "y vuelve a ejecutar g0.py.\n"
                f"Detalle: {e}"
            ) from e

        cur.execute(f"DROP TABLE IF EXISTS {TABLE};")
        cur.execute(
            f"""
            CREATE TABLE {TABLE} (
                id SERIAL PRIMARY KEY,
                frase TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM})
            );
            """
        )
    conn.commit()


def main() -> int:
    conn = None
    cursor = None
    try:
        print("[1/3] Leyendo corpus (mismo chunk que p0/c0)...")
        frases = load_sentences(CORPUS_FILE)
        print(f"      {len(frases)} frases desde '{CORPUS_FILE}'.")

        print("[2/3] Conectando y preparando esquema Pgvector...")
        conn = psycopg2.connect(**DB_CONFIG)
        ensure_pgvector_schema(conn)
        cursor = conn.cursor()
        print(f"      Extensión vector OK. Tabla '{TABLE}' creada (embedding vector({EMBEDDING_DIM})).")

        print(f"[3/3] Insertando {len(frases)} frases (commit por fila)...")
        tiempos: list[float] = []
        for i, frase in enumerate(frases, start=1):
            t0 = time.perf_counter()
            cursor.execute(
                f"INSERT INTO {TABLE} (frase) VALUES (%s);",
                (frase,),
            )
            conn.commit()
            tiempos.append(time.perf_counter() - t0)
            if i % 1000 == 0 or i == len(frases):
                print(f"      Progreso: {i}/{len(frases)}")

        print_time_metrics(tiempos, "inserción de texto (storing textual data)")
        print("\n[G0] Completado. Siguiente paso: g1.py")
        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
            print("Conexión cerrada.")


if __name__ == "__main__":
    sys.exit(main())
