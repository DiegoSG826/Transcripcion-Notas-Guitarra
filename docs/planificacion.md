# Plan del proyecto: cinco semanas laborables y ocho etapas

## 1. Alcance

**Entrada:** archivo WAV de guitarra.  
**Salida:** predicciones de notas y tablatura ASCII, además de eventos exportables a CSV/JSON y una demo web local para cargar un WAV, visualizar la tablatura y descargar sus resultados.

La propuesta técnica utiliza el audio mono de micrófono de GuitarSet, una representación CQT y un modelo CNN/CRNN. Se implementa con programación orientada a objetos: entidades de dominio, servicios de aplicación y adaptadores de archivos o presentación tienen responsabilidades separadas. La salida prevista del modelo es `lote × tiempo × 6 × 21`: seis cuerdas y 21 clases por cuerda, sujeto a la definición de etiquetas acordada en la implementación.

La primera versión se limita a WAV, notas, tablatura ASCII y una demo web local para una solicitud por vez. No incluye despliegue público, cuentas, historial persistente ni procesamiento concurrente. No se deben presentar como implementadas funciones que todavía no hayan sido desarrolladas y verificadas.

## 2. Calendario oficial del proyecto

El Planner establece el periodo del **26 de octubre al 28 de noviembre de 2026**. Este intervalo contiene cinco semanas laborables de lunes a viernes. Se seguirá este calendario comprimido; no se sustituirá por el plan anterior de siete semanas.

| Bloque | Fechas | Etapas activas | Hito de seguimiento |
|---|---|---|---|
| Semana 1 | 26–30 oct. | E01, E02 | Alcance, entorno y primer inventario de datos |
| Semana 2 | 2–6 nov. | E02, E03, E04 | Datos organizados; prototipos de preprocesamiento/CQT |
| Semana 3 | 9–13 nov. | E03, E04, E05 | Audio/CQT alineados; avance del modelo |
| Semana 4 | 16–20 nov. | E05, E06, E08 | Modelo verificado y entrenamiento piloto |
| Semana 5 | 23–27 nov. | E06, E07, E08 | Checkpoint, inferencia, evaluación y entrega integrada |
| Fecha límite final | 28 nov. (sábado) | E07, E08 | Entrega final indicada en el Planner |

El 28 de noviembre cae en sábado. Se conserva como fecha límite solicitada en el Planner para E07 y E08; las tareas que el archivo ubica antes deben cerrarse en sus fechas asignadas.

## 3. Responsables y revisores

Cada integrante es responsable de dos etapas completas. Los revisores son parte del control de calidad y no sustituyen al responsable de la etapa.

| Etapa | Nombre | Responsable | Revisor | Inicio | Fecha límite |
|---|---|---|---|---|---|
| E01 | Planeación, fundamentos y entorno | García Mendoza Diego | Del Castillo Aguirre Nancy Ashanti | 26 oct. | 30 oct. |
| E02 | Datos, anotaciones y particiones | Del Castillo Aguirre Nancy Ashanti | García Mendoza Diego | 26 oct. | 6 nov. |
| E03 | Preprocesamiento de audio | Del Castillo Aguirre Nancy Ashanti | Alba Olarra Luis Angel | 2 nov. | 13 nov. |
| E04 | Extracción CQT y alineación temporal | Alba Olarra Luis Angel | Guerra Ruiz Gabriel | 2 nov. | 13 nov. |
| E05 | Modelo CNN de referencia y CRNN | Alba Olarra Luis Angel | Del Castillo Aguirre Nancy Ashanti | 9 nov. | 20 nov. |
| E06 | Entrenamiento y aumento de datos | Guerra Ruiz Gabriel | Alba Olarra Luis Angel | 16 nov. | 27 nov. |
| E07 | Evaluación y análisis de errores | Guerra Ruiz Gabriel | García Mendoza Diego | 23 nov. | 28 nov. |
| E08 | Inferencia, suavizado y tablatura | García Mendoza Diego | Guerra Ruiz Gabriel | 16 nov. | 28 nov. |

## 4. Descripción de etapas y criterios de aceptación

### E01. Planeación, fundamentos y entorno — Diego
**Fechas:** 26–30 de octubre.  
**Entregables:** plan, glosario, diagrama de flujo, repositorio, entorno, contratos de módulos y arquitectura POO.
**Aceptación:** entorno reproducible; interfaces y responsabilidades de clases documentadas; alcance limitado a WAV, notas y tablatura ASCII.

### E02. Datos, anotaciones y particiones — Nancy Ashanti
**Fechas:** 26 de octubre–6 de noviembre.  
**Entregables:** `data_loader.py`, clases de carga/validación/partición, manifiesto, incidencias, etiquetas y particiones train/val/test.
**Aceptación:** pares WAV/JAMS verificados; cada archivo pertenece a una sola partición; intersecciones vacías.

### E03. Preprocesamiento de audio — Nancy Ashanti
**Fechas:** 2–13 de noviembre.  
**Entregables:** `preprocessing.py`, clases de preprocesamiento/segmentación, configuración, pruebas, figuras y trazabilidad temporal.
**Aceptación:** tiempos verificables; audio corto o silencioso válido; valores finitos; última ventana comprobada.

### E04. Extracción CQT y alineación temporal — Luis Angel
**Fechas:** 2–13 de noviembre.  
**Entregables:** `cqt_extraction.py`, clases de CQT/alineación/lotes, matrices, etiquetas alineadas, configuración y verificador.
**Aceptación:** dimensiones documentadas; valores finitos; tiempos alineados; guardado y carga comprobados.

### E05. Modelo CNN de referencia y CRNN — Luis Angel
**Fechas:** 9–20 de noviembre.  
**Entregables:** `model_crnn.py`, clases de fábrica/modelo CNN/CRNN, resumen, diagrama, configuración y pruebas.
**Aceptación:** salida `lote × tiempo × 6 × 21`; se conserva el eje temporal; pérdida finita en una prueba pequeña.

### E06. Entrenamiento y aumento de datos — Gabriel
**Fechas:** 16–27 de noviembre.  
**Entregables:** `train.py`, clases de entrenamiento/experimentos/checkpoints, configuraciones, logs, curvas, checkpoints y tabla de experimentos.
**Aceptación:** experimentos trazables; checkpoint recuperable; modelo elegido usando únicamente validación.

### E07. Evaluación y análisis de errores — Gabriel
**Fechas:** 23–28 de noviembre.  
**Entregables:** `evaluate.py`, clases de evaluación/métricas/reporte, seis matrices de confusión, tabla comparativa y análisis.
**Aceptación:** métricas reproducibles; relleno excluido; conjunto de prueba utilizado sin ajustar parámetros con él.

### E08. Inferencia, suavizado y tablatura — Diego
**Fechas:** 16–28 de noviembre.  
**Entregables:** `inference.py`, `tablature_generator.py`, `web_app.py`, clases de servicio/suavizado/tablatura/exportación, salida ASCII, eventos CSV/JSON y manual.
**Aceptación:** WAV externo procesado sin JAMS; seis líneas en orden correcto; silencio distinguido de traste 0; la demo local permite cargar un WAV, consultar la tablatura y descargar los archivos de esa misma ejecución.

## 5. Dependencias críticas

- **E01** establece alcance, entorno y contratos de módulos.
- **E02** proporciona los pares de audio/anotación, etiquetas y particiones para las etapas de procesamiento.
- **E03 y E04** deben entregar representaciones y alineación temporal verificables antes de cerrar el flujo de datos para el modelo.
- **E05** debe comprobar dimensiones y una prueba de aprendizaje pequeña antes del entrenamiento completo.
- **E06** depende de la aceptación de E02–E05 para el entrenamiento completo.
- **E07** requiere el modelo de E06 y los parámetros de suavizado de E08 congelados antes de evaluar el conjunto de prueba.
- **E08** requiere E03–E06 estables para inferencia real; coordina con E07 la congelación de los parámetros de suavizado. La demo web de E08 consume la misma función de inferencia que la interfaz de línea de comandos; no crea un flujo distinto.

Las etapas se solapan en el calendario, pero el solapamiento no elimina sus dependencias. Los prototipos pueden empezar con muestras validadas; la aceptación final requiere los insumos definidos en el Planner.

## 6. Reglas técnicas y de calidad

1. Registrar versiones, configuración, semillas, particiones y comandos usados.
2. Verificar pares WAV/JAMS y evitar que un mismo archivo aparezca en más de una partición.
3. Documentar la alineación temporal entre audio, CQT y etiquetas.
4. Comprobar dimensiones y valores finitos en las matrices.
5. Mantener el eje temporal de las predicciones y validar las seis salidas de 21 clases.
6. Elegir el checkpoint y ajustar el suavizado usando validación, no el conjunto de prueba.
7. Excluir relleno de las métricas cuando corresponda y documentar el método.
8. No inventar resultados: toda cifra de rendimiento debe proceder de una ejecución real.
9. Distinguir silencio de traste 0 en la representación de tablatura.
10. No marcar una etapa como terminada hasta que existan evidencias y se cumplan sus criterios de aceptación. Se utilizarán las evidencias de ejecución documentadas en `docs/validacion_paquete.md`.

## 7. Uso del equipo

El trabajo de auditoría, preprocesamiento, CQT, pruebas pequeñas e inferencia se planifica localmente. Se debe medir el tiempo de un entrenamiento piloto. Si el entrenamiento completo no resulta viable con los recursos disponibles, se podrá considerar Colab sin cambiar silenciosamente el código, la configuración ni la trazabilidad del experimento.

No se debe asumir aceleración por GPU Radeon. El uso de CPU, Colab y sus limitaciones debe documentarse según las pruebas reales.

## 8. Disponibilidad y seguimiento

El archivo Planner define fechas, responsables, revisores y criterios, pero no confirma por sí mismo la cantidad de horas que cada persona podrá dedicar. El equipo debe registrar avances y bloqueos en el sistema de seguimiento acordado.

El estado inicial de las tareas es **No iniciado**. Una tarea solo debe pasar a completada cuando el entregable exista, cumpla el criterio de aceptación y el revisor haya validado la evidencia.

## 9. Seguimiento de E01-T01: preparación del entorno

**Estado de la tarea:** completada para las comprobaciones realizadas el 26 de octubre de 2026.

Evidencias registradas en `docs/validacion_paquete.md`:

- Python 3.12.4 ejecutado desde el entorno virtual `.venv`.
- Dependencias de `requirements.txt` instaladas correctamente.
- `pip check` terminó con `No broken requirements found.`
- Importación de NumPy, SciPy, pandas, Matplotlib, scikit-learn, librosa, SoundFile, JAMS y TensorFlow correcta.
- Auditoría preliminar: 360 WAV, 360 JAMS y 360 pares únicos; estado `sin_incidencias_preliminares`.
- Informe de auditoría: `reports/auditoria_datos_20261026T055130_817529Z.json`.

La revisión de Nancy se omite como requisito de cierre de E01-T01, según el acuerdo del equipo. Esto no elimina las revisiones ni los criterios de aceptación de las demás tareas. La auditoría tampoco certifica la corrección musical ni la alineación temporal de las anotaciones; estas comprobaciones quedan pendientes de pruebas específicas.

## 10. Estructura académica sugerida

Portada; resumen y palabras clave; índice; introducción; planteamiento del problema; objetivos; justificación; marco teórico y antecedentes; metodología; implementación; experimentos y resultados; discusión; conclusiones; trabajo futuro; referencias y anexos/manual.

Los resultados que aún no se hayan obtenido deben identificarse como pendientes.

## 11. Referencias

Xi, Q., Bittner, R. M., Pauwels, J., Ye, X., & Bello, J. P. (2018). *GuitarSet: A dataset for guitar transcription*. Proceedings of the 19th International Society for Music Information Retrieval Conference, 453–460. https://archives.ismir.net/ismir2018/paper/000188.pdf

Wiggins, A., & Kim, Y. (2019). *Guitar tablature estimation with a convolutional neural network*. Proceedings of the 20th International Society for Music Information Retrieval Conference, 284–291. https://doi.org/10.5281/zenodo.3527800
