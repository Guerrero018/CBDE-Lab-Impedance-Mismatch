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
| min | | | |
| max | | | |
| avg | | | |
| std | | | |

#### Top-2 (`p2`)

| Métrica | Ambas métricas / query | Euclidiana | Coseno |
|---------|------------------------|------------|--------|
| min | | | |
| max | | | |
| avg | | | |
| std | | | |

### 1.3 Respuestas [PQ1]

**¿Son estables los tiempos de inserción de texto y embeddings?**  
En `p0`, la inserción de texto es **estable**: avg ≈ 0.64 ms y std ≈ 0.41 ms, con un max ocasional (~7 ms) atribuible a flushes/WAL. Los embeddings (`p1`) deben medirse aparte: el `UPDATE` de `REAL[384]` aumenta el payload; la generación del modelo suele dominar y variar más con la longitud de frase. Completar la tabla de `p1` tras la ejecución.

**¿Son estables los tiempos de consulta? ¿Diferencias entre métricas?**  
En `p2` cada query hace el mismo trabajo asintótico (1×`cdist` sobre n vectores). Euclidiana y coseno deberían tener latencias similares; pequeñas diferencias vienen del coste aritmético interno de SciPy. La inestabilidad, si aparece, suele deberse a GC o a cache de CPU, no a PostgreSQL (la BD ya no participa tras el `fetch`).

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
| min | | | |
| max | | | |
| avg | | | |
| std | | | |

#### Top-2 nativo (`c2`)

| Métrica | Ambas / query | Euclidiana (l2) | Coseno |
|---------|---------------|-----------------|--------|
| min | | | |
| max | | | |
| avg | | | |
| std | | | |

### 2.3 Respuestas [CQ1]

**¿Estables inserciones de texto y embeddings?**  
En `c0`, avg ≈ 39 ms y std ≈ 8.6 ms: **menos estable y ~60× más lento** que `p0` por fila, porque cada `add` actualiza el índice HNSW aunque el embedding sea placeholder. Completar `c1` tras la ejecución; la generación debería parecerse a `p1` (mismo modelo).

**¿Estables las queries? ¿Diferencias entre métricas?**  
Las queries nativas usan ANN (HNSW): latencia mucho menor que el brute-force de `p2`, y suele ser estable. Euclidiana (colección `l2`) vs coseno (colección `cosine`) pueden diferir ligeramente por la geometría del índice, no por un traslado a Python.

**¿Se pueden medir por separado texto y embeddings en Chroma?**  
**No de forma nativa:** el `add` por defecto embebe y guarda juntos. En este lab **sí** lo separamos artificialmente: `c0` con placeholders (sin modelo) y `c1` con `update` de vectores reales. Eso responde al enunciado y evidencia que la API acopla ambos pasos.

**¿Qué mejoraría Chroma?**  
Inserción por lotes (`add` multi-id), ajustar `ef` / parámetros HNSW, una sola colección si solo se necesita una métrica, o embedding function integrada si no hace falta comparar el mismo modelo externo.

---

## 3. Discusión: PostgreSQL vs Chroma (impedance mismatch)

| Aspecto | PostgreSQL (sin Pgvector) | Chroma |
|---------|---------------------------|--------|
| Modelo de datos | Relacional + `REAL[]` opaco | Colección de documentos + vectores indexados |
| Almacenar texto | Natural (`TEXT`) | Natural (`documents`), con matiz de embeddings obligatorios |
| Almacenar vectores | Mapeo a array SQL (mismatch) | Tipo de primera clase |
| Top-k similitud | En el **cliente** (mismatch alto) | En el **motor** (HNSW) |
| Código de distancias | SciPy/NumPy explícito | Casi inexistente |
| Pros | Madurez SQL, transacciones, ecosistema | Bajo mismatch vectorial, queries simples y rápidas |
| Contras | No escala el k-NN sin extensión; más código | Menos flexible para SQL analítico; una métrica/colección |

**Conclusión:** el lab ilustra que cuando la naturaleza de los datos (vectores) no coincide con el modelo interno de la BD (tablas/arrays genéricos), el coste de traducción —en tiempo, memoria y complejidad de código— es el *impedance mismatch*. Chroma reduce ese gap para embeddings; PostgreSQL sin Pgvector lo maximiza en la fase de consulta.

---

## 4. Cómo reproducir

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
```

Opcional: `python run_experiments.py` ejecuta la cadena y guarda salidas en `results/`.

---

## 5. Parte opcional (Pgvector)

No incluida en esta entrega. Si se añade (`g0`–`g2`), comparar tiempos de k-NN nativo en PostgreSQL+Pgvector frente a Chroma y discutir pros/cons (integración SQL vs motor vectorial dedicado).
