"""
c0.py — [C0] Carga de texto (bookCorpus) en ChromaDB.

Enunciado: cargar las MISMAS frases que en p0.py y medir min / max / avg / std
del tiempo de inserción de datos textuales.

Nota (relevante para CQ1):
  Chroma exige embeddings al hacer add() si embedding_function=None.
  Para medir SOLO texto (sin el modelo), insertamos documents + un embedding
  placeholder (ceros de dim 384). c1.py los sustituirá por embeddings reales.
  Así forzamos la separación texto / embeddings que la API no ofrece de forma nativa.
"""

import statistics
import sys
import time
from pathlib import Path

import chromadb

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
CORPUS_FILE = Path("bookcorpus_10k.txt")
CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "corpus"
# Dimensión de all-MiniLM-L6-v2 (la usará c1.py con embeddings reales)
EMBEDDING_DIM = 384
PLACEHOLDER_EMBEDDING = [0.0] * EMBEDDING_DIM


def print_time_metrics(times: list[float], label: str) -> None:
    """Imprime min, max, avg y std según el enunciado del laboratorio."""
    if not times:
        print(f"[AVISO] No hay mediciones para '{label}'.")
        return

    std_val = statistics.pstdev(times) if len(times) == 1 else statistics.stdev(times)
    print("\n" + "=" * 60)
    print(f"Resultados [C0]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def load_sentences(path: Path) -> list[str]:
    """Lee el corpus: una frase limpia por línea (mismo split que p0.py)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró '{path}'. Colócalo en el directorio del laboratorio."
        )
    with path.open("r", encoding="utf-8") as f:
        frases = [line.strip() for line in f if line.strip()]
    if not frases:
        raise ValueError(f"El archivo '{path}' no contiene frases.")
    return frases


def reset_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Equivalente a TRUNCATE: borra la colección si existe y la recrea.
    embedding_function=None → nosotros aportamos los vectores (placeholders / c1).
    """
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"      Colección '{COLLECTION_NAME}' eliminada (reset).")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        # Espacio por defecto para búsquedas nativas en c2 (cosine).
        # Euclidiana se podrá medir con otra colección o post-proceso en c2.
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def main() -> int:
    try:
        print("[1/3] Leyendo corpus (mismo archivo que p0.py)...")
        frases = load_sentences(CORPUS_FILE)
        print(f"      {len(frases)} frases cargadas desde '{CORPUS_FILE}'.")

        print(f"[2/3] Abriendo PersistentClient en '{CHROMA_PATH}'...")
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = reset_collection(client)
        print(f"      Colección '{COLLECTION_NAME}' lista.")

        print(
            f"[3/3] Insertando {len(frases)} documentos "
            f"(1 add por frase, embeddings placeholder)..."
        )
        tiempos_insercion: list[float] = []

        for i, frase in enumerate(frases, start=1):
            # IDs string "1".."N" alineados con SERIAL de PostgreSQL → mismos QUERY_IDS en c2/p2
            doc_id = str(i)

            t0 = time.perf_counter()
            collection.add(
                ids=[doc_id],
                documents=[frase],
                embeddings=[PLACEHOLDER_EMBEDDING],
            )
            tiempos_insercion.append(time.perf_counter() - t0)

            if i % 1000 == 0 or i == len(frases):
                print(f"      Progreso: {i}/{len(frases)}")

        print_time_metrics(
            tiempos_insercion,
            "inserción de texto en Chroma (storing textual data)",
        )
        print(
            "\nNota: los embeddings son placeholders (ceros). "
            "Ejecuta c1.py para generar y almacenar los embeddings reales."
        )
        print("\n[C0] Completado. Siguiente paso: c1.py")
        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
