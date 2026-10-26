# Sistema inteligente para la transcripción de audio de guitarra

**Instituto Politécnico Nacional · Escuela Superior de Cómputo (ESCOM)**  
**Unidad de aprendizaje:** Redes Neuronales y Aprendizaje Profundo · **Grupo:** 6BV1

**Título:** Sistema inteligente para la transcripción de audio de guitarra mediante redes neuronales convolucionales y recurrentes.

## Integrantes

- Alba Olarra Luis Angel.
- Del Castillo Aguirre Nancy Ashanti.
- García Mendoza Diego.
- Guerra Ruiz Gabriel.

## Descripción

El proyecto busca procesar una grabación WAV de guitarra, estimar notas a lo largo del tiempo y producir una tablatura ASCII básica. La propuesta técnica emplea el conjunto GuitarSet, preprocesamiento de audio, una representación CQT y un modelo CNN/CRNN.

El alcance inicial se limita a WAV, notas y tablatura ASCII, con eventos exportables a CSV/JSON. Las funcionalidades y resultados se considerarán completos únicamente después de implementarlos y verificarlos.

## Fuente de planificación

El archivo Planner es la referencia principal del calendario, las tareas, responsables, revisores, dependencias, entregables y criterios de aceptación. El proyecto se organiza en ocho etapas durante el periodo del **26 de octubre al 28 de noviembre de 2026**, que contiene cinco semanas laborables.

El detalle de etapas y criterios está en [`docs/planificacion.md`](docs/planificacion.md). Se conserva ese nombre de archivo para no romper referencias existentes; su contenido corresponde al calendario vigente del Planner.

### Responsables por etapa

| Etapa | Nombre | Responsable |
|---|---|---|
| E01 | Planeación, fundamentos y entorno | García Mendoza Diego |
| E02 | Datos, anotaciones y particiones | Del Castillo Aguirre Nancy Ashanti |
| E03 | Preprocesamiento de audio | Del Castillo Aguirre Nancy Ashanti |
| E04 | Extracción CQT y alineación temporal | Alba Olarra Luis Angel |
| E05 | Modelo CNN de referencia y CRNN | Alba Olarra Luis Angel |
| E06 | Entrenamiento y aumento de datos | Guerra Ruiz Gabriel |
| E07 | Evaluación y análisis de errores | Guerra Ruiz Gabriel |
| E08 | Inferencia, suavizado y tablatura | García Mendoza Diego |

Cada integrante es responsable de dos etapas. Los revisores y las fechas por tarea se consultan en el Planner.

## Estado actual

La auditoría preliminar ejecutada en el entorno virtual de Windows con Python 3.12.4 encontró 360 WAV, 360 archivos JAMS y 360 pares únicos. La salida fue `sin_incidencias_preliminares`. El informe generado por esa ejecución es:

```text
reports/auditoria_datos_20261026T055130_817529Z.json
```

También se verificó la instalación de `requirements.txt`, `pip check` no detectó requisitos incumplidos y las nueve bibliotecas principales pudieron importarse correctamente. El detalle de estas pruebas está en [`docs/validacion_paquete.md`](docs/validacion_paquete.md).

La auditoría comprueba el inventario y los aspectos estructurales implementados por el script; no certifica que las anotaciones sean musicalmente correctas ni que estén alineadas temporalmente con el audio. El modelo entrenado, sus métricas, la inferencia completa y la tablatura generada no deben declararse terminados hasta contar con evidencias de ejecución.

## 1. Requisitos

- Windows 10/11.
- Python 3.12.x; se recomienda 3.12.4 para reproducir el entorno inicial.
- Git.
- Conexión estable a Internet.
- Espacio libre para descargar y extraer GuitarSet y guardar datos derivados.

En Windows puede haber varias instalaciones de Python. En el equipo de referencia, `python --version` mostró Python 3.13.9 y `py -3.12 --version` mostró Python 3.12.4. Usa `py -3.12` para seleccionar explícitamente la versión prevista.

```bat
py -0p
py -3.12 --version
git --version
```

## 2. Crear y activar el entorno virtual

### CMD

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python --version
python -c "import sys; print(sys.executable)"
```

La ruta del intérprete debe apuntar a `.venv\Scripts\python.exe`.

### PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -c "import sys; print(sys.executable)"
```

No ejecutes `Activate.ps1` desde CMD. En CMD usa `activate.bat`. Si no quieres activar el entorno, invoca directamente `.venv\Scripts\python.exe`.

## 3. Instalar las dependencias

Con el entorno virtual activado:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` instala las bibliotecas Python; no descarga GuitarSet. Las dependencias deben probarse con Python 3.12 y mantenerse actualizadas conforme se implementen los módulos.

### Comprobaciones del entorno

Con el entorno virtual activo, ejecuta:

```bat
.venv\\Scripts\\python.exe --version
.venv\\Scripts\\python.exe -m pip check
.venv\\Scripts\\python.exe -c "import numpy, scipy, pandas, matplotlib, sklearn, librosa, soundfile, jams, tensorflow; print('Importaciones correctas')"
```

La ejecución documentada en `docs/validacion_paquete.md` obtuvo Python 3.12.4, `No broken requirements found.` y `Importaciones correctas`.

## 4. Descargar y organizar GuitarSet

Fuentes oficiales:

- Anotaciones: [annotation.zip](https://zenodo.org/records/3371780/files/annotation.zip?download=1)
- Audio mono de micrófono: [audio_mono-mic.zip](https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1)
- Registro del conjunto de datos: [Zenodo — GuitarSet](https://doi.org/10.5281/zenodo.3371780)

Desde la raíz del repositorio, ejecuta el script de preparación si está incorporado:

```bat
python scripts\preparar_datos.py
```

El script está diseñado para descargar los ZIP, verificar sus sumas MD5 y extraer los archivos esperados. Si todavía no se ha incorporado al repositorio, descarga y extrae manualmente los archivos:

```text
data/
├── downloads/
│   ├── annotation.zip
│   └── audio_mono-mic.zip
└── raw/
    └── guitarset/
        ├── annotation/
        │   └── *.jams
        └── audio_mono-mic/
            └── *.wav
```

Conserva los nombres originales y evita mantener varias copias de una misma extracción.

## 5. Auditar los datos

```bat
python scripts\auditar_datos.py
```

O, sin activar el entorno:

```bat
.venv\Scripts\python.exe scripts\auditar_datos.py
```

Resultado de referencia obtenido con Python 3.12.4:

```text
Estado: sin_incidencias_preliminares
WAV: 360 | JAMS: 360 | Pares únicos: 360
Python: 3.12.4
```

El auditor comprueba inventario, pares por nombre, archivos faltantes o duplicados, cabeceras WAV y lectura estructural de JAMS. El estado preliminar no certifica la corrección musical ni la alineación temporal.

## 6. Estructura del repositorio

```text
Transcripcion-Notas-Guitarra/
├── configs/                  # Configuraciones de experimentos
├── data/
│   ├── downloads/            # ZIP descargados localmente
│   ├── processed/            # Datos derivados
│   └── raw/guitarset/
│       ├── annotation/       # Anotaciones JAMS
│       └── audio_mono-mic/   # Audio WAV
├── docs/
│   ├── planificacion.md       # Calendario vigente de cinco semanas
│   ├── validacion_paquete.md  # Evidencias de validación del entorno
│   └── ...
├── models/                   # Modelos y checkpoints
├── notebooks/                # Exploración y alternativa Colab
├── outputs/                  # Notas y tablaturas
├── reports/                  # Auditorías, métricas y figuras
├── scripts/
│   ├── auditar_datos.py
│   └── preparar_datos.py
├── splits/                   # Particiones reproducibles
├── src/                      # Código fuente
├── .gitignore
├── README.md
└── requirements.txt
```

La estructura describe la organización prevista; algunos módulos se incorporarán conforme avancen las etapas.

## 7. Flujo de trabajo

El orden de implementación sigue las dependencias del Planner:

1. E01: alcance, repositorio, entorno y contratos de módulos.
2. E02: pares WAV/JAMS, etiquetas y particiones sin fuga de datos.
3. E03: preprocesamiento y trazabilidad temporal.
4. E04: CQT y alineación de etiquetas.
5. E05: CNN de referencia, CRNN y pruebas de dimensiones.
6. E06: entrenamiento, experimentos y selección de checkpoint por validación.
7. E07: métricas, matrices de confusión y análisis de errores.
8. E08: inferencia sobre WAV externo, suavizado, eventos y tablatura ASCII.

Las etapas pueden solaparse en fechas, pero el solapamiento no elimina las dependencias ni los criterios de aceptación. No se debe usar el conjunto de prueba para ajustar parámetros.

## 8. Git y archivos que no deben subirse

No se recomienda versionar el entorno virtual, ZIP, audio original, datos procesados voluminosos ni checkpoints grandes. Comprueba que `.gitignore` excluya al menos:

```gitignore
.venv/
data/downloads/
data/raw/guitarset/
data/processed/
models/
```

Revisa los informes JSON antes de publicarlos, porque pueden contener rutas locales. Los scripts, `requirements.txt`, README y documentación deben mantenerse en el repositorio.

## 9. Criterio para declarar una etapa terminada

No marques una etapa como completada hasta que:
1. Exista el entregable indicado en el Planner.
2. Se haya comprobado el criterio de aceptación.
3. La evidencia esté disponible y los resultados sean reproducibles.
4. El revisor asignado haya aceptado el entregable.

## 10. Fuentes

- Xi, Q., Bittner, R. M., Pauwels, J., Ye, X., & Bello, J. P. (2018). *GuitarSet: A dataset for guitar transcription*. Proceedings of the 19th International Society for Music Information Retrieval Conference, 453–460. https://archives.ismir.net/ismir2018/paper/000188.pdf
- TensorFlow. (s. f.). *Install TensorFlow with pip*. https://www.tensorflow.org/install/pip
