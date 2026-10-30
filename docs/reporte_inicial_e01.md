# Bloque inicial del reporte

**Proyecto:** Sistema inteligente para la transcripción de audio de guitarra mediante redes neuronales convolucionales y recurrentes  
**Etapa y tarea:** E01-T03 - Redactar fundamentos y plan de trabajo  
**Fecha:** 28 de octubre de 2026  
**Estado:** entregable documental. No contiene resultados experimentales.

## 1. Introducción

Una grabación de guitarra contiene información musical implícita en una señal acústica. Transformarla en una representación legible requiere localizar cambios a lo largo del tiempo, distinguir alturas musicales y, para una tablatura, estimar la cuerda y el traste usados. La dificultad aumenta cuando hay varias cuerdas activas, armónicos, ruido, resonancias y notas cuya misma altura puede ejecutarse en posiciones distintas.

Este proyecto propone un sistema de transcripción para archivos WAV de guitarra. El flujo previsto combina una representación tiempo-frecuencia mediante Transformada Q Constante (CQT), una red convolucional y recurrente (CNN/CRNN), y una conversión posterior de estados por cuerda a eventos y tablatura ASCII. GuitarSet aporta pares de audio y anotaciones para construir y evaluar el flujo. La implementación usa programación orientada a objetos para separar las entidades musicales y temporales de los servicios de procesamiento, los repositorios de archivos y las interfaces de consola o web.

## 2. Planteamiento del problema

A partir de un WAV mono de guitarra, se busca estimar una secuencia temporal de estados para seis cuerdas. Cada estado debe distinguir entre ausencia de nota y un traste de 0 a 19. Una predicción de altura no basta por sí sola: la misma nota puede aparecer en más de una cuerda, por lo que la salida representa posiciones de ejecución.

La tarea no se reduce a analizar un instante aislado. La señal evoluciona después del ataque, puede contener acordes y superposición de armónicos, y las predicciones por frame pueden mostrar cambios breves no musicales. Por ello, el sistema requiere alineación explícita entre audio, CQT y etiquetas, una arquitectura que conserve el eje temporal y una etapa de suavizado evaluada con validación.

## 3. Objetivos

### Objetivo general

Diseñar e integrar un flujo reproducible que convierta una grabación WAV de guitarra en estados por cuerda, eventos temporales y tablatura ASCII básica, usando representaciones CQT y un modelo CNN/CRNN.

### Objetivos específicos

1. Documentar el alcance, las convenciones de etiquetas, la arquitectura orientada a objetos y las interfaces entre módulos antes de implementar el flujo completo.
2. Inventariar y emparejar los WAV y JAMS de GuitarSet, interpretar sus eventos y crear particiones sin cruces.
3. Preparar audio y CQT manteniendo una referencia temporal verificable con las etiquetas.
4. Construir una CNN de referencia y una CRNN cuya salida conserve el tiempo y tenga seis distribuciones de 21 clases por frame.
5. Entrenar y elegir un checkpoint exclusivamente con evidencia del conjunto de validación.
6. Evaluar el modelo sobre el conjunto de prueba una vez congeladas las decisiones de preprocesamiento, modelo y suavizado.
7. Convertir predicciones de un WAV externo en eventos CSV/JSON y tablatura ASCII, sin requerir JAMS durante inferencia.

## 4. Justificación

La transcripción automática de guitarra concentra problemas reales de procesamiento digital de señales, reconocimiento de patrones y modelado secuencial. A diferencia de una clasificación de audio general, la tablatura exige representar la posición de ejecución y el tiempo, no solo una etiqueta musical global.

El proyecto permite aplicar estos conceptos a un conjunto de datos especializado y con anotaciones musicales estructuradas. También obliga a separar adecuadamente entrenamiento, validación y prueba, registrar configuraciones y no presentar ejemplos didácticos ni prototipos como resultados experimentales. El resultado buscado es una demostración trazable de un flujo de aprendizaje profundo, no una afirmación de transcripción perfecta.

## 5. Alcance y límites

La versión inicial acepta WAV de guitarra y genera:

- estados por frame para seis cuerdas;
- eventos con cuerda, traste, nota, inicio y final en CSV/JSON;
- tablatura ASCII de seis líneas.
- una demo web local para cargar un WAV, ver la tablatura y descargar los archivos producidos.

Se entrenará con el audio mono de micrófono y las anotaciones JAMS de GuitarSet. Cada cuerda usa 21 clases: código 0 para ausencia de nota y códigos 1 a 20 para los trastes 0 a 19. Traste 0, es decir una cuerda al aire, no equivale a silencio.

Quedan fuera de alcance los nombres de acordes, MusicXML, ejecución en tiempo real, otros instrumentos, trastes fuera de 0 a 19, despliegue público, cuentas de usuario, historial persistente y cargas concurrentes. La demo web usa la misma inferencia local de E08; no crea un modelo ni una ruta de procesamiento distinta. Un evento fuera de rango o una altura MIDI no entera debe registrarse y tratarse mediante una regla explícita; no se recorta ni redondea de forma silenciosa. La agrupación de estados tampoco detecta por sí sola ataques repetidos en un mismo traste, por lo que esta limitación debe declararse en la salida si no se incorpora un detector específico.

## 6. Metodología y plan de trabajo

El calendario vigente dura cinco semanas laborables, del 26 de octubre al 28 de noviembre de 2026. La planificación organiza el trabajo en ocho etapas:

| Etapa | Propósito | Resultado esperado |
|---|---|---|
| E01 | Planeación, fundamentos y entorno | Documentación, entorno reproducible y contratos |
| E02 | Datos, anotaciones y particiones | Pares WAV/JAMS, etiquetas y particiones disjuntas |
| E03 | Preprocesamiento | Audio y ventanas con trazabilidad temporal |
| E04 | CQT y alineación | Matrices CQT, etiquetas y máscara de relleno |
| E05 | Modelo | CNN/CRNN con salida temporal de seis cuerdas |
| E06 | Entrenamiento | Experimentos, checkpoint y registros |
| E07 | Evaluación | Métricas, matrices de confusión y análisis |
| E08 | Inferencia y tablatura | Eventos, tablatura y demo web local para un WAV externo |

Las etapas pueden solaparse, pero no se omiten sus dependencias: E02 precede al procesamiento final; E03 y E04 alimentan al modelo; E05 precede al entrenamiento; E06 y los parámetros de suavizado congelados preceden a la prueba de E07. Las entidades de dominio y las responsabilidades de las clases se describen en [arquitectura_poo.md](arquitectura_poo.md). El detalle de fechas, responsables y criterios está en [planificacion.md](planificacion.md), y los formatos de integración en [contratos_modulos.md](contratos_modulos.md).

## 7. Estado de la evidencia

Al cierre de esta tarea se han documentado el entorno de Python, sus dependencias, el inventario preliminar de 360 WAV, 360 JAMS y 360 pares únicos, junto con los contratos de módulos. Estas evidencias no demuestran todavía que las anotaciones sean musicalmente correctas, que estén alineadas con el audio, ni que exista un modelo entrenado.

No se incluyen métricas de rendimiento porque todavía no se ha ejecutado el entrenamiento ni la evaluación. Esas cifras deben incluir configuración, partición, comando y archivo fuente cuando las produzcan E06 y E07.

## 8. Fundamentos y términos clave

La explicación didáctica de nota, acorde, armónicos, MIDI, afinación, cuerda al aire, traste, CQT, CNN, RNN, CRNN, validación, prueba e inferencia está en [fundamentos_musicales_01.md](fundamentos_musicales_01.md). Ese documento incluye ejemplos calculados y preguntas de comprensión; no contiene predicciones del modelo.

## 9. Referencias

- Xi, Q., Bittner, R. M., Pauwels, J., Ye, X., & Bello, J. P. (2018). *GuitarSet: A dataset for guitar transcription*. Proceedings of the 19th International Society for Music Information Retrieval Conference, 453-460. https://archives.ismir.net/ismir2018/paper/000188.pdf
- Wiggins, A., & Kim, Y. (2019). *Guitar tablature estimation with a convolutional neural network*. Proceedings of the 20th International Society for Music Information Retrieval Conference, 284-291. https://doi.org/10.5281/zenodo.3527800

