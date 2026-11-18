# Bitácora y revisión inicial

## 17 de noviembre de 2026: tratamiento de incidencias E02

Se implementó la política solicitada de enmascarar intervalos ambiguos por
cuerda. Se conservan 360 pistas, 62 476 eventos, las 3311 advertencias y las
particiones 240/60/60. Los 74 intervalos MIDI ambiguos de 55 pistas no se
convierten en silencio ni se usan como objetivos de entrenamiento/evaluación.

Se verificaron las 360 pistas sobre tiempos del manifiesto: 981 posiciones de
frame/cuerda excluidas con hop_length=512, ninguna posición adicional por
conflictos. También se comprobó una CQT real `[1687,84]` con 22 posiciones
excluidas. La suite completa obtuvo 68 pruebas correctas, incluidas pérdida y
gradientes TensorFlow. No se ejecutó entrenamiento de un modelo.

Comandos, política, alcance y evidencia: [politica_etiquetas_e02.md](politica_etiquetas_e02.md).
No se registra una revisión musical manual ni una aceptación de terceros. Las
etapas posteriores deben integrar la pérdida y las métricas con estas máscaras.
La lista siguiente conserva el seguimiento inicial como registro histórico.

## Estado conocido

- [x] Tema, institución, materia, cuatro integrantes y grupo definidos.
- [x] Entrada por archivos grabados y salida inicial notas/tablatura de texto.
- [x] Audio de micrófono mostrado en captura: ZIP con 360 elementos WAV visibles.
- [ ] Extracción local e integridad de todos los WAV comprobadas.
- [ ] Descarga y extracción de `annotation.zip` confirmadas.
- [ ] Correspondencia audio–anotación comprobada en la computadora del equipo.
- [ ] Semántica, cuerdas y tiempos de las anotaciones inspeccionados.
- [ ] Tres horas de disponibilidad aclaradas: frecuencia y por persona/equipo.
- [ ] Fechas de los tres avances confirmadas.
- [ ] CPU, Python y formato del reporte registrados.

Una captura no verifica los bytes del dataset ni demuestra que un modelo funcione.
El auditor también tiene un alcance limitado: no valida la corrección musical.

## Primera sesión propuesta de tres horas

| Tiempo orientativo | Actividad | Producto |
|---|---|---|
| 45 minutos | Leer fundamentos y discutir cinco preguntas. | Conceptos compartidos y dudas registradas. |
| 60 minutos | Extraer audios y anotaciones conservando nombres. | Carpetas listas. |
| 45 minutos | Ejecutar auditor y revisar incidencias. | JSON y resumen del entorno. |
| 30 minutos | Registrar decisiones y repartir siguiente tarea. | Bitácora actualizada. |

Es una distribución de una sesión, no una afirmación sobre la disponibilidad semanal.
La descarga o extracción podría tardar más según conexión y almacenamiento.

## Plantilla de sesión

- Fecha: [PENDIENTE].
- Participantes: [PENDIENTE].
- Objetivo: [PENDIENTE].
- Qué se ejecutó y comando exacto: [PENDIENTE].
- Archivos de evidencia: [PENDIENTE].
- Resultado observado: [RESULTADO PENDIENTE].
- Problemas y decisión tomada: [PENDIENTE].
- Próxima tarea, responsable y fecha: [PENDIENTE].

## Plantilla de experimento posterior

| Campo | Valor |
|---|---|
| Identificador | [PENDIENTE] |
| Pregunta que responde | [PENDIENTE] |
| Versión del código / commit | [PENDIENTE] |
| Versión del dataset y revisión de incidencias | [PENDIENTE] |
| Archivos de particiones | [PENDIENTE] |
| Configuración y semilla | [PENDIENTE] |
| Equipo y versiones | [PENDIENTE] |
| Tiempo y modelo guardado | [RESULTADO PENDIENTE] |
| Métricas y ruta del informe | [RESULTADO PENDIENTE] |
| Interpretación y limitaciones | [RESULTADO PENDIENTE] |

## Antes de entrenar

- [ ] Orden de las cuerdas verificado con ejemplos.
- [ ] Traste 0 y sin nota tienen códigos diferentes.
- [ ] Casos fuera del rango 0–19 documentados.
- [ ] CQT y etiquetas alineadas visualmente.
- [ ] Particiones guardadas antes de formar ventanas.
- [ ] Estadísticas aprendidas solo de entrenamiento.
- [ ] Un lote tiene dimensiones correctas y valores finitos.
- [ ] La red puede aprender una muestra pequeña.
- [ ] Guardar/cargar produce predicciones equivalentes dentro de tolerancia definida.

## Antes de entregar

- [ ] Las instrucciones se comprobaron en un entorno preparado desde cero.
- [ ] Los resultados incluyen configuración y partición.
- [ ] Se reportan errores y rendimiento de clases activas, no solo accuracy.
- [ ] Los ajustes de suavizado se eligieron con validación.
- [ ] Las figuras tienen títulos, ejes, unidades y explicación.
- [ ] Los ejemplos didácticos están separados de resultados experimentales.
- [ ] Las conclusiones responden a los objetivos con evidencia.
- [ ] Las referencias corresponden a fuentes realmente consultadas.
- [ ] El usuario conoce las limitaciones de la tablatura y de ataques repetidos.
