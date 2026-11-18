# Política E02: intervalos ambiguos por cuerda

Fecha: 17 de noviembre de 2026. Política: `enmascarar_intervalos_ambiguos_por_cuerda_v1`.

## Decisión y alcance

Se conservan las 360 pistas y los 62 476 eventos exportados por T02. Los 3311
eventos con advertencia siguen teniendo una etiqueta normalizada. Los 74 eventos
`MIDI_AMBIGUO`, distribuidos en 55 pistas, permanecen en la fuente de eventos con
`label_class: null` y su estado original `INVALIDO`. Al alinearlos, se desactiva
solo la supervisión de su cuerda en `[onset_s, offset_s)`.

Esto trata la incertidumbre para entrenamiento y evaluación; no corrige la
anotación musical ni afirma que la nota original sea incorrecta. Los WAV, JAMS,
informes históricos y particiones originales permanecen como fuentes.

Los umbrales se mantienen: desviación MIDI <= 0.25 sin aviso; (0.25, 0.49) con
aviso; >= 0.49 sin objetivo discreto. La misma regla se aplica a train, validation
y test. No se ajustaron umbrales a partir de resultados del conjunto de prueba.

## Fuente única y trazabilidad

`configs/etiquetas.json` identifica la política, los umbrales y cinco entradas:

- Manifiesto: `reports/e02_t01/e02_t01_manifiesto.json`.
- Eventos completos: `reports/e02_t02_normalizado/e02_t02_eventos.json`.
- Incidencias: `reports/e02_t02_normalizado/e02_t02_incidencias.json`.
- Resumen: `reports/e02_t02_normalizado/e02_t02_resumen.json`.
- Particiones: `splits/particiones.json`.

`CatalogoEtiquetas.cargar(raiz)` comprueba coherencia entre esos archivos,
etiquetas contra los umbrales, cobertura de incidencias, ausencia de duplicados,
tiempos y separación de pistas/intérpretes. Guarda SHA-256 de la configuración
y de cada fuente en los artefactos derivados. No depende del `data_root` absoluto
del manifiesto para leer los informes, por lo que se puede usar en otro equipo.

Solo `MIDI_AMBIGUO` con intervalo y cuerda válidos es una incidencia tratable
automáticamente por esta política. Cuerdas faltantes, archivos ilegibles,
intervalos inválidos, errores fuera de rango u otras incidencias detienen la
preparación. También se rechaza quitar un ambiguo de los eventos mientras su
incidencia siga registrada. Un conjunto vacío de notas por sí solo no demuestra
que exista silencio: el catálogo exige los informes completos de T02.

Los estados históricos `pistas_para_revision` y `apto_entrenamiento` siguen
representando la validación estricta sin máscaras. El nuevo informe distingue
`pistas_utilizables_con_mascaras`: no reclasifica esos eventos como notas válidas.

## Contrato temporal y máscaras

Para cada centro de frame dentro del intervalo ambiguo:

```text
labels[t, cuerda] = -1
label_valid_mask[t, cuerda] = False
valid_mask[t] = True                 # sigue siendo audio real
```

Las otras cinco cuerdas conservan sus etiquetas y su supervisión, salvo que
tengan una exclusión propia. En un intervalo conocido sin nota, `labels=0` y
la supervisión es válida. La cuerda al aire sigue siendo clase 1.

El relleno tiene `valid_mask=False` y todas sus cuerdas se excluyen. Una posición
de aprendizaje válida se define mediante:

```python
supervision_mask = valid_mask[..., None] & label_valid_mask
```

`ConstructorLotes` devuelve esa máscara junto con `x`, `y`, `valid_mask` y
`label_valid_mask`. Las etiquetas son -1 o 0..20; un -1 sin exclusión explícita
produce un error. Los solapamientos contradictorios entre notas de una cuerda
también se excluyen, con independencia del orden de los eventos. La verificación
del conjunto actual no encontró posiciones adicionales por esos conflictos.

Para un segmento se usa `catalogo.alinear_pista(track_id, cqt.frame_center_s,
inicio_s=segmento.start_s, fin_valido_s=segmento.end_s)`. Los tiempos CQT de
entrada son locales al segmento; para exportar `frame_center_s` respecto al WAV,
se les suma `segmento.start_s`. Los frames en el relleno o en el instante exacto
del fin real no se supervisan. La máscara se aplica al centro del frame, según
el contrato, sin ampliar el intervalo por el soporte temporal del filtro CQT.

## Pérdida y métricas para E05–E07

`supervision.py` aporta funciones utilizables con el futuro modelo:

```python
from transcripcion_guitarra.supervision import (
    entropia_cruzada_enmascarada, contar_aciertos_enmascarados,
)

# Dentro de GradientTape, con salida [B,T,6,21]:
loss = entropia_cruzada_enmascarada(
    lote['y'], probabilidades, lote['valid_mask'], lote['label_valid_mask'],
)
# Para logits, especificar from_logits=True.

conteos = contar_aciertos_enmascarados(
    lote['y'], clases_predichas, lote['valid_mask'], lote['label_valid_mask'],
)
```

La pérdida selecciona las posiciones supervisadas ANTES de la entropía cruzada
y divide por su cantidad, no por el tamaño del tensor rellenado. Los excluidos
no producen gradientes. No basta con reemplazar -1 por 0 ni con multiplicar
una pérdida ya calculada por cero. Un lote sin supervisión devuelve pérdida
cero; el entrenador debe omitir su paso de optimización, incluidos regularización
y actualización de estado del optimizador.

Las métricas suman aciertos y soporte entre lotes; no promedian exactitudes de
lotes de diferentes tamaños. Si una cuerda no tiene soporte, su exactitud es
`None`. Las matrices de confusión y otras métricas deberán seleccionar con
la misma máscara. Los excluidos no son aciertos, errores ni verdaderos silencios.
Se debe reportar cuántas posiciones se excluyeron al publicar resultados.

Estas funciones y sus pruebas no equivalen a un entrenamiento del modelo. El
entrenador y evaluador de las etapas posteriores todavía deben integrarlas.

## Reproducir la verificación

Desde la raíz del repositorio, con Python del entorno preparado:

```powershell
.\.venv\Scripts\python.exe scripts/preparar_mascaras_etiquetas.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Se generan:

- `reports/e02_mascaras/e02_mascaras_verificacion.json`: procedencia, resumen,
  resultado por pista y lista de los 74 intervalos.
- `reports/e02_mascaras/e02_intervalos_enmascarados.csv`: lista legible de los
  intervalos, pista, partición, cuerda, tiempo, MIDI y tratamiento. El índice
  `indice_evento_exportado` es cero-basado en la lista JSON de eventos; no es el
  `event_index` de la anotación JAMS.

Con hop_length=512 y las frecuencias del manifiesto se verificaron las 360 pistas
y los 74 intervalos: 981 posiciones de frame/cuerda excluidas por ambigüedad,
cero exclusiones adicionales por conflictos, y el resto de etiquetas idéntico
a la referencia sin ambiguos. Es una verificación sobre tiempos del manifiesto,
no una extracción CQT de todo el dataset. El conteo depende de la rejilla y debe
regenerarse si cambian sample_rate, hop_length o la fuente de eventos.

Las particiones se conservan: 240/60/60 pistas; los intervalos se distribuyen
45/4/25 en train/validation/test. No se descartaron las 55 pistas afectadas.

Para extraer y guardar un ejemplo real, con sus máscaras y procedencia:

```powershell
.\.venv\Scripts\python.exe scripts/probar_cqt.py data/raw/guitarset/audio_mono-mic/00_BN1-147-Gb_comp_mic.wav --output data/processed/e02_mascaras/ejemplo_ambiguo.npz
```

El NPZ contiene CQT, etiquetas, tiempos y las tres máscaras; los datos derivados
en `data/processed/` no se versionan. La demo consume el catálogo por defecto.
Un WAV desconocido falla salvo que se indique `--sin-etiquetas`; ese modo
desactiva toda supervisión y no fabrica etiquetas de silencio.

Para incluir la comprobación de esa matriz real en el informe JSON:

```powershell
.\.venv\Scripts\python.exe scripts/preparar_mascaras_etiquetas.py --demo-npz data/processed/e02_mascaras/ejemplo_ambiguo.npz
```

La ejecución del 17 de noviembre obtuvo CQT `[1687,84]`, etiquetas `[1687,6]`
y 22 posiciones excluidas de la cuarta cuerda, con el resto de etiquetas intacto.
El campo `demo_cqt` registra la forma y SHA-256 del NPZ comprobado. El comando
sin `--demo-npz` genera únicamente la verificación de la rejilla del manifiesto.
La suite completa terminó con **68 pruebas correctas**, incluidas selección
por cuerda, límites temporales, relleno, pérdida diferenciable, gradientes cero
en excluidos y lote sin supervisión.

Para regenerar T02 en otro equipo se puede indicar la ubicación local sin editar
el manifiesto histórico:

```powershell
.\.venv\Scripts\python.exe scripts/validar_etiquetas_guitarset.py --data-root data/raw/guitarset
```

La salida predeterminada ahora es `reports/e02_t02_normalizado`. Regenerar T02
cambia sus fuentes; después se deben repetir la verificación, las pruebas y la
generación de matrices derivadas. Los informes exploratorio e inicial se
conservan como históricos y no son la fuente de entrenamiento de esta política.

## Cierre y siguiente integración

La política de tratamiento fue solicitada por el usuario. Las incidencias
continúan registradas; no se documenta una revisión musical manual ni una
aceptación de terceros que no haya ocurrido. El cierre académico de la etapa
y la revisión del equipo se registran por separado. E03/E04 deben completar su
validación de audio/CQT real y E05–E07 deben consumir las máscaras descritas.
