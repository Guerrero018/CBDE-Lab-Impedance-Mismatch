# Setup Pgvector (Windows / PostgreSQL 17)

La parte opcional `[G0]–[G2]` requiere la extensión **pgvector** en el servidor PostgreSQL.

## Opción A — Binario precompilado (rápido)

1. Descarga el zip para PG 17, por ejemplo:  
   https://github.com/andreiramani/pgvector_pgsql_windows/releases
2. Copia:
   - `vector.dll` → `C:\Program Files\PostgreSQL\17\lib\`
   - `vector.control` y `vector--*.sql` → `C:\Program Files\PostgreSQL\17\share\extension\`
3. Reinicia el servicio PostgreSQL (servicios de Windows).
4. Verifica:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

## Opción B — Compilar desde fuente

Con *x64 Native Tools Command Prompt* (Visual Studio), como administrador:

```bat
set "PGROOT=C:\Program Files\PostgreSQL\17"
cd %TEMP%
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

Docs oficiales: https://github.com/pgvector/pgvector

## Ejecutar la parte opcional

```bash
python g0.py   # texto → tabla corpus_pgvector
python g1.py   # embeddings vector(384) + índices HNSW
python g2.py   # top-2 nativo (<-> y <=>)
```

Usa el mismo `bookcorpus_10k.txt` y los mismos `QUERY_IDS` que p2/c2.
