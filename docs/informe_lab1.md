# Lab 1 — Impedance Mismatch on Vector Data

**Asignatura:** CBDE (UPC)  
**Repositorio:** https://github.com/Guerrero018/CBDE-Lab-Impedance-Mismatch  
**Corpus:** [`bookcorpus_10k.txt`](../bookcorpus_10k.txt) (10.000 frases; generado con `prepare_corpus.py` desde HuggingFace `SamuelYang/bookcorpus`, derivado de BookCorpus)  
**Modelo:** `sentence-transformers/all-MiniLM-L6-v2`  
**Scripts:** `p0.py`–`p2.py` (PostgreSQL), `c0.py`–`c2.py` (Chroma)

---

## 1. PostgreSQL

### 1.1 Decisiones de diseño (impedance mismatch)

| Decisión | Motivación |
|----------|------------|
| Tabla `corpus(id SERIAL, frase TEXT, embedding REAL[])` | Texto en tipo nativo; vectores como array SQL genérico **sin Pgvector** (restricción del lab). |
| Sin índice vectorial | PostgreSQL no ofrece ANN nativo sin extensiones; cualquier vecino se calcula fuera. |
| `p0`: `INSERT` + `COMMIT` por fila | Mide el coste real de persistencia textual (I/O + WAL), comparable a un `add` unitario en Chroma. |
| `p1`: embeddings generados en Python y `UPDATE` a `REAL[]` | El modelo vive en la aplicación; la BD solo almacena floats. Eso es el mapeo vector→array. |
| `p1`: tiempos de **generación** y **almacenamiento** separados | El enunciado pide *storing embeddings*; separar evita confundir inferencia del modelo con el mismatch de almacenamiento. |
| `p2`: `SELECT` de todos los embeddings → NumPy/SciPy | Sin Pgvector no hay k-NN en SQL; hay que materializar ~10k×384 floats en el cliente (**máximo impedance mismatch**). |
| Métricas: Euclidiana + similitud del coseno | Coherentes con la teoría del lab; SciPy `cdist`. |
| `QUERY_IDS` fijos `[1,1111,…,9999]` | Reproducibilidad y misma muestra en Chroma (`c2`). |

**Impacto en código/llamadas:** cada sentencia es una ida y vuelta a la BD; en `p2` hay una extracción masiva + O(n) distancias por consulta en Python. El número de líneas y de llamadas crece porque la lógica vectorial no vive en el motor.

### 1.2 Resultados experimentales

> Completar tras ejecutar `python p0.py && python p1.py && python p2.py`.  
> Los valores se imprimen al final de cada script (min / max / avg / std).

#### Inserción de texto (`p0`)

| Métrica | Valor (s) |
|---------|-----------|
| min | 0.000256 |
| max | 0.006966 |
| avg | 0.000640 |
| std | 0.000409 |

#### Almacenamiento de embeddings (`p1` — UPDATE)

| Métrica | Generación | Almacenamiento (UPDATE) | Ciclo total |
|---------|------------|---------------------------|-------------|
| min | 0.010094 | 0.001858 | 0.012183 |
| max | 1.180173 | 1.077214 | 1.209745 |
| avg | 0.025478 | 0.003626 | 0.029104 |
| std | 0.021706 | 0.011395 | 0.025146 |

#### Top-2 (`p2`)

| Métrica | Ambas métricas / query | Euclidiana | Coseno |
|---------|------------------------|------------|--------|
| min | 0.016386 | 0.005660 | 0.009973 |
| max | 0.031977 | 0.016714 | 0.014719 |
| avg | 0.019518 | 0.007826 | 0.011435 |
| std | 0.004621 | 0.003192 | 0.001537 |

*(Fetch de los 10k embeddings a Python: 2.999 s, una sola vez.)*

### 1.3 Respuestas [PQ1]

**¿Son estables los tiempos de inserción de texto y embeddings?**  
En `p0`, la inserción de texto es **estable**: avg ≈ 0.64 ms y std ≈ 0.41 ms, con un max ocasional (~7 ms) atribuible a flushes/WAL. Los embeddings (`p1`) deben medirse aparte: el `UPDATE` de `REAL[384]` aumenta el payload; la generación del modelo suele dominar y variar más con la longitud de frase. Completar la tabla de `p1` tras la ejecución.

**¿Son estables los tiempos de consulta? ¿Diferencias entre métricas?**  
Sí, relativamente estables (ambas métricas: avg ≈ 19.5 ms, std ≈ 4.6 ms). Euclidiana (avg ≈ 7.8 ms) fue algo más rápida que coseno (avg ≈ 11.4 ms) en SciPy; la diferencia es pequeña frente al coste dominante del *impedance mismatch*: el `fetch` único de ~3 s para materializar los 10k×384 vectores en Python antes de calcular nada.

**¿Qué mejoraría el rendimiento en PostgreSQL sin Pgvector?**  
- `COPY` / inserts por lotes (menos round-trips).  
- Arrays en `BYTEA` o tablas normalizadas `(id, dim, value)` solo si ayuda al análisis (no al k-NN).  
- Materializar distancias no es viable a escala.  
- Funciones PL/Python en el servidor aún pagarían el mismatch y complican el despliegue.  
La mejora real de vecinos cercanos exige un índice vectorial (Pgvector u otro motor nativo).

---

## 2. Chroma

### 2.1 Decisiones de diseño (impedance mismatch)

| Decisión | Motivación |
|----------|------------|
| `PersistentClient(path=./chroma_db)` | Persistencia local comparable a PostgreSQL en disco. |
| Colección `corpus` con `embedding_function=None` | Controlamos el modelo (mismo que en PostgreSQL) y medimos almacenamiento aparte. |
| `c0`: documentos + embedding **placeholder** (ceros 384-d) | Chroma exige vectores si no hay EF; así medimos inserción de texto **sin** llamar al modelo ([CQ1]). |
| `c1`: `update` con embeddings reales | Separación texto/embeddings forzada, alineada con `p0`/`p1`. |
| `hnsw:space=cosine` en `corpus` | Búsqueda nativa por coseno. |
| Colección auxiliar `corpus_l2` en `c2` | Chroma fija **una** métrica por colección; la réplica L2 permite Euclidiana nativa. |
| Mismos `QUERY_IDS` e IDs `"1"…"N"` | Comparación justa con `p2`. |
| `query(query_embeddings=…, n_results=3)` + excluir self | k-NN sobre HNSW **dentro** de Chroma (bajo mismatch). |

**Impacto:** menos código de distancias; la aplicación pide vecinos y Chroma responde. El setup de `corpus_l2` es un coste de ingeniería para soportar dos métricas, no del path crítico de cada query.

### 2.2 Resultados experimentales

> Completar tras `python c0.py && python c1.py && python c2.py`.

#### Inserción de texto (`c0`)

| Métrica | Valor (s) |
|---------|-----------|
| min | 0.026514 |
| max | 0.331069 |
| avg | 0.039344 |
| std | 0.008593 |

#### Almacenamiento de embeddings (`c1` — update)

| Métrica | Generación | Almacenamiento (update) | Ciclo total |
|---------|------------|---------------------------|-------------|
| min | 0.012875 | 0.027309 | 0.042570 |
| max | 0.790946 | 1.005162 | 1.026780 |
| avg | 0.040191 | 0.050682 | 0.090873 |
| std | 0.031354 | 0.032916 | 0.048634 |

#### Top-2 nativo (`c2`)

| Métrica | Ambas / query | Euclidiana (l2) | Coseno |
|---------|---------------|-----------------|--------|
| min | 0.007016 | 0.003470 | 0.003394 |
| max | 0.030647 | 0.025547 | 0.005770 |
| avg | 0.010631 | 0.006200 | 0.004183 |
| std | 0.007079 | 0.006803 | 0.000705 |

*(Setup de la colección L2 auxiliar: 15.657 s, no incluido en las métricas de query.)*

### 2.3 Respuestas [CQ1]

**¿Estables inserciones de texto y embeddings?**  
En `c0`, avg ≈ 39 ms y std ≈ 8.6 ms: **menos estable y ~60× más lento** que `p0` por fila, porque cada `add` actualiza el índice HNSW aunque el embedding sea placeholder. En `c1`, el **almacenamiento** (avg ≈ 51 ms) supera a la generación (avg ≈ 40 ms): actualizar el índice HNSW con vectores reales es más caro que el `UPDATE REAL[]` de PostgreSQL (avg ≈ 3.6 ms). La generación es del mismo orden que en `p1` (mismo modelo).

**¿Estables las queries? ¿Diferencias entre métricas?**  
Las queries nativas son más rápidas que el brute-force de `p2` (ambas métricas: avg ≈ 10.6 ms vs ≈ 19.5 ms en PostgreSQL+Python; y **sin** el fetch de ~3 s). El coseno (avg ≈ 4.2 ms, std ≈ 0.7 ms) fue más estable que L2 (avg ≈ 6.2 ms, std ≈ 6.8 ms, con algún max más alto). No hay traslado masivo de vectores al cliente: el índice HNSW absorbe el k-NN.

**¿Se pueden medir por separado texto y embeddings en Chroma?**  
**No de forma nativa:** el `add` por defecto embebe y guarda juntos. En este lab **sí** lo separamos artificialmente: `c0` con placeholders (sin modelo) y `c1` con `update` de vectores reales. Eso responde al enunciado y evidencia que la API acopla ambos pasos.

**¿Qué mejoraría Chroma?**  
Inserción por lotes (`add` multi-id), ajustar `ef` / parámetros HNSW, una sola colección si solo se necesita una métrica, o embedding function integrada si no hace falta comparar el mismo modelo externo.

---

## 3. Pgvector (opcional)

### 3.1 Decisiones de diseño (impedance mismatch)

| Decisión | Motivación |
|----------|------------|
| Tabla propia `corpus_pgvector` | No pisa `corpus` (`REAL[]`) de la parte obligatoria; comparación limpia. |
| Tipo `vector(384)` | Representación nativa de embeddings (cierra el mismatch del array SQL). |
| Mismo corpus, modelo y `QUERY_IDS` | Comparación justa con p*/c*. |
| Operadores `<->` (L2) y `<=>` (cosine) | Mismas dos métricas; k-NN **en el servidor**. |
| Índices HNSW (`vector_l2_ops` + `vector_cosine_ops`) tras `g1` | Aceleran top-2 nativo en `g2` (análogo al índice de Chroma). |
| Scripts `g0`/`g1`/`g2` espejo de p0/p1/p2 | Mismas métricas min/max/avg/std exigidas por el enunciado. |

### 3.2 Resultados experimentales

> Completar tras instalar Pgvector (`docs/pgvector_setup.md` / `install_pgvector_windows.ps1` como Admin) y ejecutar `g0.py`, `g1.py`, `g2.py`.

#### Inserción de texto (`g0`)

| Métrica | Valor (s) |
|---------|-----------|
| min | |
| max | |
| avg | |
| std | |

#### Almacenamiento de embeddings (`g1` — UPDATE vector)

| Métrica | Generación | Almacenamiento (UPDATE) | Ciclo total |
|---------|------------|---------------------------|-------------|
| min | | | |
| max | | | |
| avg | | | |
| std | | | |

#### Top-2 nativo (`g2`)

| Métrica | Ambas / query | Euclidiana (`<->`) | Coseno (`<=>`) |
|---------|---------------|--------------------|----------------|
| min | | | |
| max | | | |
| avg | | | |
| std | | | |

### 3.3 Conclusiones Pgvector vs Chroma

| | Pgvector | Chroma |
|--|----------|--------|
| Pros | SQL + transacciones + JOINs; vectores en el mismo motor que datos relacionales; operadores e índices documentados | API simple orientada a embeddings; persistencia local ligera; menos SQL que aprender |
| Contras | Hay que instalar extensión en el servidor; tuning HNSW en SQL; Windows no trae binario oficial | Menos expresividad relacional; una métrica por colección (aquí resolvimos con `corpus_l2`) |
| Impedance mismatch | Bajo para vectores (tipo + índice nativos), manteniendo el modelo relacional | Bajo para vectores; motor dedicado |

---

## 4. Discusión: PostgreSQL vs Chroma vs Pgvector

| Aspecto | PostgreSQL (sin Pgvector) | Chroma | Pgvector |
|---------|---------------------------|--------|----------|
| Modelo de datos | Relacional + `REAL[]` opaco | Colección + vectores indexados | Relacional + `vector(N)` |
| Almacenar texto | Natural (`TEXT`) | Natural (`documents`) | Natural (`TEXT`) |
| Almacenar vectores | Array SQL (mismatch) | Tipo de primera clase | Tipo nativo extensión |
| Top-k similitud | Cliente (SciPy) | Motor (HNSW) | Motor (`<->` / `<=>` + HNSW) |
| Código de distancias | Explícito en Python | Mínimo | SQL corto |
| Pros | Madurez SQL | Bajo mismatch, API simple | SQL + vectores juntos |
| Contras | k-NN no escala | Menos SQL analítico | Setup de extensión |

**Conclusión:** sin Pgvector, PostgreSQL maximiza el mismatch en consulta. Chroma y Pgvector lo reducen con índices nativos; Chroma es un motor vectorial dedicado, Pgvector acerca ese modelo al ecosistema SQL.

---

## 5. Cómo reproducir

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
pip install -r requirements.txt

python prepare_corpus.py   # genera bookcorpus_10k.txt

# PostgreSQL (BD vector_lab, tabla corpus ya creada)
python p0.py
python p1.py   # costoso en CPU (10k embeddings)
python p2.py

# Chroma
python c0.py
python c1.py
python c2.py

# Pgvector (opcional)
# Admin: .\install_pgvector_windows.ps1  → reiniciar servicio → CREATE EXTENSION vector;
python g0.py
python g1.py
python g2.py
```

Opcional: `python run_experiments.py` ejecuta la cadena y guarda salidas en `results/`.
