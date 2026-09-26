"""
c1.py — [C1] Generación y almacenamiento de embeddings en ChromaDB.

Enunciado: conectar a Chroma, generar embeddings por frase y almacenarlos.
Medir min / max / avg / std del tiempo de almacenamiento de embeddings.

Se miden por separado (como en p1.py / útil para CQ1):
  - generación del embedding (modelo, en Python)
  - almacenamiento (update en Chroma)  ← métrica principal del enunciado

Requisito: haber ejecutado c0.py (documentos + placeholders).
"""

import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import chromadb
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuración (misma colección / path que c0.py; mismo modelo que p1.py)
# ---------------------------------------------------------------------------
CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "corpus"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling ponderado por attention mask (recomendado por Sentence-Transformers)."""
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def embed_phrase(tokenizer, model, phrase: str) -> List[float]:
    """Embedding L2-normalizado (mismo pipeline que p1.py → comparables)."""
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
    print(f"Resultados [C1]: {label}")
    print("=" * 60)
    print(f"  Operaciones : {len(times)}")
    print(f"  min         : {min(times):.6f} s")
    print(f"  max         : {max(times):.6f} s")
    print(f"  avg         : {statistics.mean(times):.6f} s")
    print(f"  std         : {std_val:.6f} s")
    print("=" * 60)


def fetch_documents(collection: chromadb.Collection) -> List[Tuple[str, str]]:
    """
    Recupera (id, document) insertados por c0.py.
    Ordena por id numérico para alinear con el orden de PostgreSQL.
    """
    result = collection.get(include=["documents"])
    ids = result.get("ids") or []
    documents = result.get("documents") or []

    if not ids:
        return []

    pairs = list(zip(ids, documents))
    # ids son "1".."N" (strings); orden numérico estable
    pairs.sort(key=lambda x: int(x[0]))
    return pairs


def store_embedding(collection: chromadb.Collection, doc_id: str, embedding: List[float]) -> None:
    """Sustituye el placeholder de c0 por el embedding real (update nativo)."""
    collection.update(ids=[doc_id], embeddings=[embedding])


def main() -> int:
    try:
        print(f"[1/4] Abriendo PersistentClient en '{CHROMA_PATH}'...")
        if not CHROMA_PATH.exists():
            print(
                f"[ERROR] No existe '{CHROMA_PATH}'. Ejecuta primero c0.py.",
                file=sys.stderr,
            )
            return 1

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )
        print(f"      Colección '{COLLECTION_NAME}' OK (n={collection.count()}).")

        print(f"[2/4] Cargando modelo '{MODEL_NAME}'...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        print("      Modelo cargado.")

        print("[3/4] Recuperando documentos de Chroma...")
        rows = fetch_documents(collection)
        if not rows:
            print(
                "[ERROR] Colección vacía. Ejecuta primero c0.py.",
                file=sys.stderr,
            )
            return 1
        print(f"      {len(rows)} documentos encontrados.")

        print("[4/4] Generando embeddings y actualizándolos en Chroma...")
        times_gen: List[float] = []
        times_store: List[float] = []

        for i, (doc_id, frase) in enumerate(rows, start=1):
            if frase is None:
                print(f"[AVISO] Documento id={doc_id} sin texto; se omite.")
                continue

            t_gen0 = time.perf_counter()
            embedding = embed_phrase(tokenizer, model, frase)
            times_gen.append(time.perf_counter() - t_gen0)

            t_store0 = time.perf_counter()
            store_embedding(collection, doc_id, embedding)
            times_store.append(time.perf_counter() - t_store0)

            if i % 500 == 0 or i == len(rows):
                print(
                    f"      Progreso: {i}/{len(rows)} | "
                    f"gen={times_gen[-1]:.4f}s store={times_store[-1]:.4f}s"
                )

        print_time_metrics(times_gen, "generación de embeddings (modelo)")
        print_time_metrics(
            times_store,
            "almacenamiento de embeddings / update Chroma (storing embeddings)",
        )

        times_total = [g + s for g, s in zip(times_gen, times_store)]
        print_time_metrics(times_total, "ciclo completo (generación + almacenamiento)")

        print("\n[C1] Completado. Siguiente paso: c2.py")
        return 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
