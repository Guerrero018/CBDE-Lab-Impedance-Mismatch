"""
prepare_corpus.py — Genera bookcorpus_10k.txt (~10k frases limpias).

Intenta, en orden:
  1) Dataset Parquet ligero compatible con BookCorpus en HuggingFace.
  2) Descarga del tar histórico de BookCorpus (pesado, ~1 GB).
  3) Fallback: textos de dominio público (Gutenberg) tokenizados en frases.
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

OUT_FILE = Path("bookcorpus_10k.txt")
CACHE_DIR = Path(".cache_bookcorpus")
TAR_NAME = "bookcorpus.tar.bz2"
URL = "https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2"
TARGET = 10_000

# Datasets HF en formato Parquet (sin scripts deprecados)
HF_CANDIDATES = [
    ("bookcorpus/bookcorpus", "train"),  # puede fallar si solo hay script
    ("SamuelYang/bookcorpus", "train"),
]

GUTENBERG_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/11/11-0.txt",  # Alice
    "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock
    "https://www.gutenberg.org/files/98/98-0.txt",  # Tale of Two Cities
]


def clean_line(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text: str) -> list[str]:
    # Segmentación simple suficiente para el lab (no es un parser lingüístico).
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [clean_line(p) for p in parts]


def collect(sentences: list[str], seen: set[str], candidates: list[str], target: int) -> bool:
    for line in candidates:
        if len(line) < 20 or len(line) > 500:
            continue
        if line in seen:
            continue
        seen.add(line)
        sentences.append(line)
        if len(sentences) % 1000 == 0:
            print(f"  {len(sentences)}/{target}")
        if len(sentences) >= target:
            return True
    return False


def try_huggingface(target: int) -> list[str] | None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[AVISO] 'datasets' no instalado; se omite HF.")
        return None

    for name, split in HF_CANDIDATES:
        print(f"Probando HuggingFace: {name} (streaming)...")
        try:
            ds = load_dataset(name, split=split, streaming=True)
            sentences: list[str] = []
            seen: set[str] = set()
            for row in ds:
                text = clean_line(str(row.get("text", row.get("sentence", ""))))
                if not text:
                    continue
                # Algunas filas son párrafos: re-segmentamos
                chunk = split_sentences(text) if len(text) > 200 else [text]
                if collect(sentences, seen, chunk, target):
                    print(f"OK vía {name}")
                    return sentences
            if sentences:
                print(f"OK parcial vía {name}: {len(sentences)}")
                return sentences
        except Exception as e:
            print(f"  Falló {name}: {e}")
    return None


def try_tar(target: int) -> list[str] | None:
    dest = CACHE_DIR / TAR_NAME
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not (dest.exists() and dest.stat().st_size > 1_000_000):
            print(f"Descargando tar BookCorpus (~1 GB):\n  {URL}")

            def _progress(block_num: int, block_size: int, total_size: int) -> None:
                if block_num % 500 != 0:
                    return
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100.0, 100.0 * downloaded / total_size)
                    print(f"  {pct:5.1f}%", flush=True)

            urllib.request.urlretrieve(URL, dest, reporthook=_progress)
        else:
            print(f"Usando caché {dest}")

        sentences: list[str] = []
        seen: set[str] = set()
        with tarfile.open(dest, "r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                with io.TextIOWrapper(f, encoding="utf-8", errors="ignore") as text:
                    for raw in text:
                        line = clean_line(raw)
                        if collect(sentences, seen, [line], target):
                            return sentences
        return sentences or None
    except Exception as e:
        print(f"Tar BookCorpus falló: {e}")
        return None


def try_gutenberg(target: int) -> list[str]:
    print("Fallback: frases desde Project Gutenberg (dominio público)...")
    sentences: list[str] = []
    seen: set[str] = set()
    for url in GUTENBERG_URLS:
        print(f"  Descargando {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"  Error: {e}")
            continue
        # Quitar cabecera/pie típicos de Gutenberg de forma burda
        body = re.split(r"\*\*\* START OF .+ \*\*\*", raw, maxsplit=1)
        text = body[1] if len(body) > 1 else raw
        body = re.split(r"\*\*\* END OF .+ \*\*\*", text, maxsplit=1)
        text = body[0]
        if collect(sentences, seen, split_sentences(text), target):
            break
    return sentences


def main() -> int:
    sentences = try_huggingface(TARGET)
    if sentences is None or len(sentences) < TARGET:
        got = try_tar(TARGET)
        if got:
            sentences = got

    if sentences is None or len(sentences) < TARGET // 2:
        sentences = try_gutenberg(TARGET)

    if not sentences:
        print("[ERROR] No se pudo construir el corpus.", file=sys.stderr)
        return 1

    if len(sentences) < TARGET:
        print(
            f"[AVISO] Solo {len(sentences)} frases (objetivo {TARGET}).",
            file=sys.stderr,
        )

    OUT_FILE.write_text("\n".join(sentences[:TARGET]) + "\n", encoding="utf-8")
    print(f"Escrito '{OUT_FILE}' con {min(len(sentences), TARGET)} frases.")
    print(
        "Nota: documenta en el informe la fuente efectiva "
        "(BookCorpus HF/tar o Gutenberg fallback)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
