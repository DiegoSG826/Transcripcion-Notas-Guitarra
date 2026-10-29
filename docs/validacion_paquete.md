# Validación del entorno y del paquete de datos

**Proyecto:** Sistema inteligente para la transcripción de audio de guitarra mediante redes neuronales convolucionales y recurrentes  
**Etapa:** E01 — Planeación, fundamentos y entorno  
**Tarea:** E01-T01 — Preparar repositorio y entorno Python  
**Plataforma:** Windows  
**Python:** 3.12.4  
**Fecha de ejecución:** 26 de octubre de 2026

## 1. Objetivo

Documentar las comprobaciones ejecutadas en el equipo de desarrollo para preparar el entorno virtual, instalar las dependencias del proyecto, comprobar las importaciones principales y ejecutar la auditoría preliminar de los datos.

## 2. Entorno virtual

Se activó el entorno virtual desde la raíz del repositorio:

```bat
.venv\Scripts\activate.bat
```

Se verificó el intérprete con:

```bat
.venv\Scripts\python.exe --version
```

Salida:

```text
Python 3.12.4
```

Para las pruebas descritas en este documento se invocó directamente el intérprete de `.venv\Scripts\python.exe`, de modo que los comandos no dependieran de la versión global de Python seleccionada por Windows.

## 3. Actualización de pip

Comando ejecutado:

```bat
.venv\Scripts\python.exe -m pip install --upgrade pip
```

Resultado: la actualización terminó correctamente y pip pasó de la versión 24.0 a la 26.2.2.

## 4. Instalación de dependencias

Comando ejecutado:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Resultado final observado:

```text
Successfully installed ...
```

La instalación terminó correctamente e incluyó las bibliotecas principales del proyecto, entre ellas NumPy, SciPy, pandas, Matplotlib, scikit-learn, librosa, SoundFile, JAMS y TensorFlow. La versión de TensorFlow instalada fue **2.19.1**.

## 5. Comprobación de dependencias

Comando ejecutado:

```bat
.venv\Scripts\python.exe -m pip check
```

Salida:

```text
No broken requirements found.
```

No se detectaron requisitos incumplidos mediante `pip check`.

## 6. Prueba de importación

Comando ejecutado:

```bat
.venv\Scripts\python.exe -c "import numpy, scipy, pandas, matplotlib, sklearn, librosa, soundfile, jams, tensorflow; print('Importaciones correctas')"
```

Salida final:

```text
Importaciones correctas
```

Durante la ejecución TensorFlow mostró mensajes informativos relacionados con oneDNN. No impidieron que la prueba terminara correctamente.

Esta comprobación confirma que las nueve bibliotecas importadas pudieron cargarse en el entorno actual. No sustituye las pruebas funcionales que se realizarán cuando se implementen los módulos de procesamiento y entrenamiento.

## 7. Auditoría preliminar de datos

Comando ejecutado:

```bat
.venv\Scripts\python.exe scripts\auditar_datos.py
```

Salida obtenida:

```text
Estado: sin_incidencias_preliminares
WAV: 360 | JAMS: 360 | Pares únicos: 360
Python: 3.12.4
Informe: reports/auditoria_datos_20261026T055130_817529Z.json
Esta revisión no certifica la corrección musical ni la alineación temporal.
```

El script encontró 360 archivos WAV, 360 archivos JAMS y 360 pares únicos según sus criterios de auditoría. El informe JSON generado se conserva en:

`reports/auditoria_datos_20261026T055130_817529Z.json`

El estado `sin_incidencias_preliminares` se limita a las comprobaciones implementadas en el auditor. No certifica la corrección musical de las anotaciones ni su alineación temporal con el audio.

## 8. Resumen de resultados

| Comprobación | Resultado |
|---|---|
| Activación del entorno virtual | Correcta |
| Versión de Python | 3.12.4 |
| Actualización de pip | Correcta; versión final 26.2.2 |
| Instalación de `requirements.txt` | Correcta |
| `pip check` | Sin requisitos incumplidos detectados |
| Importación de las nueve bibliotecas | Correcta |
| Estado de auditoría preliminar | `sin_incidencias_preliminares` |
| Archivos WAV | 360 |
| Archivos JAMS | 360 |
| Pares únicos | 360 |

## 9. Conclusión y alcance de la validación

Las pruebas ejecutadas permiten confirmar que el entorno virtual está preparado para continuar con el desarrollo inicial: las dependencias se instalaron, `pip check` no reportó requisitos incumplidos y las bibliotecas principales pudieron importarse.

La auditoría también identificó 360 archivos WAV, 360 archivos JAMS y 360 pares únicos sin incidencias preliminares según los criterios del script. Aún se requieren pruebas específicas para verificar la integridad del contenido, la validez musical de las anotaciones y su alineación temporal.

Por lo tanto, estas evidencias respaldan el cierre de la preparación del entorno de E01-T01, pero no certifican el funcionamiento del modelo CNN/CRNN ni la validez musical completa del conjunto de datos.

## 10. Evidencias relacionadas

- `requirements.txt`: declaración de dependencias.
- `scripts/auditar_datos.py`: script de auditoría.
- `reports/auditoria_datos_20261026T055130_817529Z.json`: informe generado por la ejecución documentada.
- Registro de comandos y salidas ejecutadas el 26 de octubre de 2026.

## 11. E01-T02: alcance, flujo e interfaces

El 28 de octubre de 2026 se documentó el contrato de integración en
[`contratos_modulos.md`](contratos_modulos.md). Define el alcance limitado a WAV,
notas y tablatura ASCII; el flujo E02--E08; la correspondencia entre fuente JAMS,
cuerda convencional y eje de tensor; las 21 clases por cuerda; el uso de máscara de
relleno; y la convención temporal de los frames.

La decisión de codificación es: clase `0` para ausencia de nota y clases `1` a `20`
para trastes `0` a `19`. En particular, traste 0 se mantiene distinto de silencio.
El documento establece además que ninguna etiqueta MIDI continua se redondea ni los
trastes fuera del rango se recortan sin una política trazable de E02.

La evidencia de esta tarea es documental y de diseño. La implementación y las pruebas
funcionales de cada interfaz quedan bajo responsabilidad de E02--E08. Conforme al
acuerdo indicado para este trabajo, no se requiere una revisión independiente para
cerrar E01-T02.

## 12. E01-T03: fundamentos y plan de trabajo

El bloque inicial del reporte se consolidó en
[reporte_inicial_e01.md](reporte_inicial_e01.md), con introducción, planteamiento,
objetivos, justificación, alcance, límites y metodología planificada. El glosario y los
ejemplos didácticos se mantienen en [fundamentos_musicales_01.md](fundamentos_musicales_01.md).

No se declaran resultados de entrenamiento, evaluación o inferencia: las evidencias
disponibles se limitan al entorno, el inventario preliminar de datos y el diseño de
interfaces. Conforme al acuerdo indicado para este trabajo, no se requiere una revisión
independiente para cerrar E01-T03.

## 13. Ajuste de E01-T02: demo web local

El contrato de módulos se actualizó a la versión 1.1 para incorporar una demo local como integración de E08. La página recibirá un solo WAV por solicitud, mostrará la tablatura ASCII y permitirá descargar los eventos CSV/JSON y el texto generado. Usará la misma inferencia de E08, no una segunda ruta de procesamiento.

No se incluye despliegue público, cuentas, historial persistente ni cargas concurrentes. La demostración sigue pendiente de implementación y prueba funcional cuando E08 tenga un checkpoint y sus módulos estables.
