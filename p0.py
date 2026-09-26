"""
p0.py — [PO] Carga de texto (bookCorpus) en PostgreSQL.

Enunciado: cargar las frases en PostgreSQL y medir min / max / avg / std
del tiempo de inserción de datos textuales (sin embeddings).
Sin Pgvector.
"""

import statistics
import sys
import time
from pathlib import Path

import psycopg2

# ---------------------------------------------------------------------------
# Configuración (misma BD que p1.py)
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vector_lab",
    "user": "postgres",
    "password": "postgres",
}
CORPUS_FILE = Path("bookcorpus_10k.txt")


def print_time_metrics(times: list[float], label: str) -> None:
    """Imprime min, max, avg y std según el enunciado del laboratorio."""
    if not times:
        print(f"[AVISO] No hay mediciones para '{label}'.")
        return

    std_val = statistics.pstdev(times) if len(times) == 1 else statistics.stdev(times)
    print("\n" + "=" * 60)
    print(f"Resultados [P0]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def load_sentences(path: Path) -> list[str]:
    """Lee el corpus: una frase limpia por línea."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró '{path}'. Colócalo en el directorio del laboratorio."
        )
    with path.open("r", encoding="utf-8") as f:
        frases = [line.strip() for line in f if line.strip()]
    if not frases:
        raise ValueError(f"El archivo '{path}' no contiene frases.")
    return frases


def main() -> int:
    conn = None
    cursor = None
    try:
        print("[1/3] Leyendo corpus...")
        frases = load_sentences(CORPUS_FILE)
        print(f"      {len(frases)} frases cargadas desde '{CORPUS_FILE}'.")

        print("[2/3] Conectando a PostgreSQL...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        # Estado limpio y reproducible antes de medir inserciones
        cursor.execute("TRUNCATE TABLE corpus RESTART IDENTITY;")
        conn.commit()
        print("      Conexión OK. Tabla corpus vaciada.")

        print(f"[3/3] Insertando {len(frases)} frases (commit por fila)...")
        tiempos_insercion: list[float] = []

        for i, frase in enumerate(frases, start=1):
            t0 = time.perf_counter()
            cursor.execute("INSERT INTO corpus (frase) VALUES (%s);", (frase,))
            # Commit por inserción: mide el coste real de persistencia (I/O + WAL)
            conn.commit()
            tiempos_insercion.append(time.perf_counter() - t0)

            if i % 1000 == 0 or i == len(frases):
                print(f"      Progreso: {i}/{len(frases)}")

        # Métrica exigida por el enunciado: storing the textual data
        print_time_metrics(tiempos_insercion, "inserción de texto (storing textual data)")
        print("\n[P0] Completado. Siguiente paso: p1.py")
        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Inesperado: {e}", file=sys.stderr)
        return 1
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
            print("Conexión cerrada.")


if __name__ == "__main__":
    sys.exit(main())
