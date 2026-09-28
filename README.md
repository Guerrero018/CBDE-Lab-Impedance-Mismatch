# CBDE Lab 1 — Impedance Mismatch (Vector Data)

Comparación **PostgreSQL** (sin Pgvector) vs **Chroma** al almacenar y consultar embeddings.

## Repo

https://github.com/Guerrero018/CBDE-Lab-Impedance-Mismatch

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `prepare_corpus.py` | Genera `bookcorpus_10k.txt` |
| `p0.py` / `p1.py` / `p2.py` | PostgreSQL: texto, embeddings, top-2 |
| `c0.py` / `c1.py` / `c2.py` | Chroma: texto, embeddings, top-2 nativo |
| `g0.py` / `g1.py` / `g2.py` | Pgvector (opcional): texto, `vector(384)`, top-2 nativo |
| `run_experiments.py` | Runner que guarda logs en `results/` |
| `docs/informe_lab1.md` | Informe del laboratorio |
| `docs/uso_IA.md` | Declaración de uso de IA (≤1 pág.) |
| `docs/pgvector_setup.md` | Cómo instalar Pgvector en Windows |
| `install_pgvector_windows.ps1` | Copia binarios pgvector (ejecutar como Admin) |
| `requirements.txt` | Dependencias Python |

## Setup rápido

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt
python prepare_corpus.py
```

PostgreSQL: BD `vector_lab`, usuario/contraseña `postgres`, tabla:

```sql
CREATE TABLE corpus (
  id SERIAL PRIMARY KEY,
  frase TEXT NOT NULL,
  embedding REAL[]
);
```

```bash
python p0.py && python p1.py && python p2.py
python c0.py && python c1.py && python c2.py

# Pgvector (opcional; requiere extensión instalada — ver docs/pgvector_setup.md)
# PowerShell Admin: .\install_pgvector_windows.ps1
python g0.py && python g1.py && python g2.py
# o:
python run_experiments.py
```

`p1`, `c1` y `g1` generan 10k embeddings con `all-MiniLM-L6-v2` (pueden tardar horas en CPU).
