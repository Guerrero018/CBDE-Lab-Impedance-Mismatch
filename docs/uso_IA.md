# Uso de herramientas de IA en el laboratorio

**Extensión:** ≤ 1 página (síntesis, sin volcar el historial completo de prompts).

## Objetivo de la asistencia

Se utilizó un asistente de código (Cursor / agente) para acelerar la implementación de los seis scripts exigidos (`p0`–`p2`, `c0`–`c2`) y alinearlos con el enunciado de *impedance mismatch* sobre datos vectoriales, no para sustituir la comprensión del problema.

## Racional de las instrucciones

1. **Contexto primero:** se proporcionó el enunciado (objetivos, restricciones sin Pgvector, modelo `all-MiniLM-L6-v2`, corpus ~10k, métricas min/max/avg/std) y el esquema PostgreSQL ya existente.
2. **Entrega incremental:** se pidió generar un script cada vez (empezando por `p1`, luego revisión de `p0`, `p2`, y la serie Chroma), validando coherencia entre pasos antes de continuar.
3. **Contraste explícito PG ↔ Chroma:** se instruyó replicar la misma lógica y los mismos `QUERY_IDS` para que la comparación de tiempos fuera justa.
4. **Separación de costes:** se pidió distinguir generación del modelo vs almacenamiento, y en Chroma forzar la separación texto/embeddings (placeholders + `update`) para poder responder [CQ1].
5. **Revisión contra el PDF del lab:** tras tener `p0`/`p1`, se reenvió el enunciado completo para corregir desvíos (p. ej. medir *storing embeddings*, no solo el ciclo mezclado).

## Refinado

Las instrucciones se endurecieron cuando aparecieron huecos: ausencia de `p0`, dataset no versionado, token accidental en `.gitignore`, y la limitación de Chroma de una métrica por colección (solución: `corpus_l2` en `c2`). También se unificó cronometría (`perf_counter`) y el formato de métricas exigido por el lab.

## Validación del output

- Lectura cruzada de cada script respecto a [PO]/[P1]/[P2]/[C0]–[C2].  
- Comprobación de que no se usa Pgvector en la parte obligatoria.  
- Verificación de misma fuente de corpus, mismo modelo y mismos IDs de consulta.  
- Ejecución local de la cadena experimental y volcado de tiempos en el informe (tablas de la §1.2 / §2.2).  
- Revisión humana del discurso de impedance mismatch en el documento (no se aceptó texto genérico sin anclarlo a decisiones de código).

## Qué no delegamos en la IA

La interpretación conceptual del mismatch, las respuestas reflexivas a [PQ1]/[CQ1]/la discusión final y la preparación del examen individual de laboratorio.
