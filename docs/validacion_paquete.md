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
