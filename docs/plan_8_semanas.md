# Plan de ocho semanas y tres avances

## Alcance confirmado y pendientes

Entrada: archivo WAV grabado. Salida: notas y tablatura ASCII básica.
Propuesta técnica: audio de micrófono de GuitarSet, CQT, CNN y capa recurrente
bidireccional, con seis salidas de 21 clases. La arquitectura concreta y los
parámetros se justificarán y revisarán tras un experimento pequeño.

La primera versión se delimita provisionalmente a guitarra de seis cuerdas, afinación
estándar y sin otros instrumentos. Esta delimitación es una propuesta técnica que
debe confirmarse, no una propiedad comprobada de grabaciones externas.

Se necesita confirmar: fechas exactas, disponibilidad por persona, modelo de CPU,
versión actual de Python, archivo de anotaciones, formato editable del reporte y
nombre del profesor. Estos puntos no impiden organizar los datos y aprender conceptos.

## Disponibilidad

- Si son 3 horas por persona por semana: 4 × 3 × 8 = 96 horas de trabajo del equipo.
- Si son 3 horas del equipo completo por semana: 3 × 8 = 24 horas-persona.
- Si son 3 horas diarias u otra frecuencia, recalcular antes de asignar tareas.

Los tiempos de ejecución automática del entrenamiento deben presupuestarse aparte;
no equivalen a horas de trabajo humano. Aún no se ha medido su duración.

## Cronograma

| Semana | Foco y actividades | Evidencia de cierre |
|---|---|---|
| 1 | Fundamentos musicales; alcance; organización de archivos; auditoría preliminar. | Datos localizados, anotaciones disponibles y bitácora iniciada. |
| 2 | Inspección de JAMS; extracción de etiquetas; CQT y alineación temporal en ejemplos. | Una grabación visualizada con sus notas de referencia. |
| 3 | Particiones por intérprete; objetivos, metodología y prototipo de lectura. | Avance 1: datos preparados y explicación del proceso. |
| 4 | CNN de referencia; CRNN pequeña; dimensiones y entrenamiento de comprobación. | Seis salidas válidas y aprendizaje en pocos ejemplos. |
| 5 | Entrenamiento inicial completo; medición de tiempo y memoria; ajustes con validación. | Curvas, configuración y modelo guardados. |
| 6 | Comparación controlada y selección del modelo; prueba de inferencia. | Avance 2: modelo entrenado y resultados de validación reales. |
| 7 | Evaluación final; suavizado; estados a eventos; tablatura; errores. | Métricas de prueba y demostración completa con limitaciones. |
| 8 | Reporte, manual, presentación, repetición de comandos y ensayo. | Avance 3/entrega final reproducible. |

Las entregas en semanas 3, 6 y 8 son propuestas hasta que el profesor indique fechas.
La escritura del reporte ocurre durante todas las semanas.

## Contenido de cada avance

### Avance 1: problema, datos y representación

- Portada con cuatro integrantes, contexto, problema, objetivos y alcance.
- Fundamentos musicales y de audio necesarios para explicar el proyecto.
- Procedencia de GuitarSet y nombres de archivos descargados.
- Inventario y revisión de emparejamiento WAV–JAMS.
- Ejemplo de anotaciones, señal y CQT alineadas.
- Particiones documentadas y plan de entrenamiento.
- Evidencias auténticas; resultados del modelo todavía pendientes.

### Avance 2: arquitectura y aprendizaje

- Arquitectura CNN de referencia y CRNN.
- Dimensiones de entrada, secuencia y seis salidas.
- Configuración de entrenamiento y criterio de selección con validación.
- Curvas de aprendizaje y resultados reales de validación.
- Guardado y carga del modelo comprobados.
- Demostración preliminar sobre un WAV.

### Avance 3: sistema completo y entrega

- Evaluación de prueba con configuración congelada.
- Métricas por cuerda, matriz de confusión y análisis de clases activas.
- Comparación antes/después del suavizado sin ajustarlo sobre prueba.
- Notas y tablatura de texto con tiempos y limitaciones documentadas.
- Reporte, anexos, manual, requisitos probados y presentación con notas del orador.

## Experimentos mínimos y extensiones

Núcleo: referencia de silencio para contextualizar accuracy, CNN sencilla, CRNN y
comparación de la CRNN con/sin suavizado. La elección de sus parámetros se realiza
con entrenamiento y validación. Una mejora nunca se da por supuesta.

Tras completar el núcleo se podrá evaluar aumento de datos. No es requisito probar
a la vez Bi-LSTM y Bi-GRU; se elegirá primero una variante. La identificación de
nombres de acordes y MusicXML queda fuera de la primera entrega funcional.

## Roles propuestos, aún sin asignar nombres

| Rol | Responsabilidad | Colaboración |
|---|---|---|
| Datos y audio | Auditar, leer anotaciones, CQT y alineación. | Revisión cruzada de etiquetas con modelado. |
| Modelado | CNN, CRNN y configuración de entrenamiento. | Compartir explicación de dimensiones con todos. |
| Integración | Inferencia, notas, tablatura y ejecución local/Colab. | Comprobar compatibilidad con entradas del modelo. |
| Evaluación y redacción | Métricas, trazabilidad, reporte y presentación. | Cada integrante documenta y revisa su contribución. |

## Criterios técnicos

1. Separar por intérprete antes de crear ventanas; nunca mezclar fragmentos de una
   misma grabación entre conjuntos. Propuesta inicial: 4/1/1 intérpretes. Esto evalúa
   generalización de intérprete, no independencia de composiciones musicales.
2. Verificar orden de cuerdas, afinación, clases y casos fuera de rango. No recortar
   silenciosamente una etiqueta al traste máximo.
3. CQT y anotaciones deben compartir una línea de tiempo conocida.
4. Calcular estadísticas aprendidas del preprocesamiento solamente con entrenamiento.
5. Examinar las clases sin nota: accuracy sola puede ocultar errores musicales.
6. Suavizar probabilidades o clases con un procedimiento justificado. No aplicar una
   mediana numérica indiscriminada sobre códigos de clase.
7. Reconocer la limitación de ataques repetidos en el mismo traste. Agrupar estados
   consecutivos no resuelve por sí mismo la detección de nuevas pulsaciones.
8. La reducción de ruido y los aumentos se incorporan solo si se justifican y evalúan.
9. Guardar configuración, particiones, semilla, versiones y modelo de cada experimento.
10. No prometer porcentajes ni tiempos antes de medirlos.

## Uso del equipo

Desarrollo, auditoría, CQT, pruebas pequeñas e inferencia se planifican localmente.
Se medirá la velocidad de un entrenamiento piloto en CPU. Si la proyección del
entrenamiento completo excede el tiempo disponible, se utilizará Colab con el mismo
código y configuración, sujeto al recurso que esté disponible en ese momento.

No cargar de una vez todas las secuencias ni duplicar arrays innecesariamente en RAM.
La capacidad real depende del modelo de CPU, parámetros CQT, duración de secuencias
y tamaño del lote. Los 16 GB declarados no garantizan cualquier configuración.

## Estructura académica

Portada; resumen/abstract y palabras clave; índice; introducción; planteamiento del
problema; objetivos; justificación; marco teórico y antecedentes; metodología;
implementación; experimentos y resultados; discusión; conclusiones; trabajo futuro;
referencias APA; anexos y manual. El documento editable queda pendiente de elección.

Todos los resultados no ejecutados se marcan con `[RESULTADO PENDIENTE]`.

## Referencias

Xi, Q., Bittner, R. M., Pauwels, J., Ye, X., & Bello, J. P. (2018). GuitarSet: A
dataset for guitar transcription. En *Proceedings of the 19th International Society
for Music Information Retrieval Conference* (pp. 453–460).
https://archives.ismir.net/ismir2018/paper/000188.pdf

Wiggins, A., & Kim, Y. (2019). Guitar tablature estimation with a convolutional neural
network. En *Proceedings of the 20th International Society for Music Information
Retrieval Conference* (pp. 284–291). https://doi.org/10.5281/zenodo.3527800
