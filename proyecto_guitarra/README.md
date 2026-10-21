# Proyecto de transcripción de guitarra: paquete inicial

Instituto Politécnico Nacional · Escuela Superior de Cómputo  
Unidad de aprendizaje: Redes Neuronales y Aprendizaje Profundo · Grupo 6BV1

**Título:** Sistema inteligente para la transcripción de audio de guitarra mediante redes neuronales convolucionales y recurrentes.

**Integrantes:**

- Alba Olarra Luis Angel.
- Del Castillo Aguirre Nancy Ashanti.
- García Mendoza Diego.
- Guerra Ruiz Gabriel.

## Estado del proyecto

Paquete inicial de planificación, fundamentos y comprobación preliminar de archivos.
Todavía no contiene una red entrenada ni resultados experimentales. El modelo, el
entrenamiento, la evaluación y la inferencia se incorporarán en las siguientes etapas.
El auditor incluido utiliza solamente la biblioteca estándar de Python: no necesitas
instalar TensorFlow, librosa ni JAMS para esta comprobación inicial.

## Decisiones confirmadas

- Windows con 16 GB de RAM y gráficos Radeon integrados.
- Ejecución local prioritaria; Colab como alternativa para entrenamiento.
- TensorFlow/Keras como biblioteca elegida para la implementación posterior.
- Procesamiento de archivos WAV grabados.
- Primera salida: notas y tablatura de texto. Nombres de acordes y MusicXML quedan
  para una ampliación posterior.
- Ocho semanas de referencia y tres entregas parciales propuestas; faltan fechas exactas.
- Experiencia suficiente en Python y Git; fundamentos musicales desde cero.
- Disponibilidad declarada: tres horas. Falta confirmar si son semanales por persona
  o del equipo completo. El plan no presenta esa interpretación como confirmada.

## Primeros pasos en Windows

1. Extrae este paquete en una ubicación que elijas. Como ejemplo usaremos
   `C:\Proyectos\proyecto_guitarra`; la ruta es una propuesta, no una carpeta ya creada
   en tu computadora.
2. Abre PowerShell y entra en esa carpeta:

   ```powershell
   # Sustituye la ruta si elegiste otra ubicación.
   cd C:\Proyectos\proyecto_guitarra
   # Comprueba qué Python tienes disponible.
   py --version
   # Crea las carpetas de trabajo sin borrar ni mover archivos existentes.
   py scripts\auditar_datos.py --preparar
   ```

3. Extrae los WAV de `audio_mono-mic.zip` en
   `data\raw\guitarset\audio_mono-mic`.
4. Extrae los archivos `.jams` de `annotation.zip` en
   `data\raw\guitarset\annotation`.
5. Conserva nombres originales. Si el ZIP contiene una carpeta adicional, el auditor
   también buscará dentro de ella. No coloques varias copias de una misma extracción.
6. Ejecuta:

   ```powershell
   # Revisa nombres, pares audio/anotación, JSON y cabeceras WAV.
   py scripts\auditar_datos.py
   ```

7. Revisa el resumen y comparte el archivo JSON generado en `reports`.

Si `py` no existe pero `python --version` funciona, sustituye `py` por `python`.
No es necesario crear un entorno virtual para ejecutar este auditor sin dependencias.
Prepararemos el entorno aislado y las versiones de ML en la siguiente etapa.

### ¿Qué comprueba el auditor?

- Cuenta WAV y JAMS y busca pares por identificador.
- Elimina solamente el sufijo `_mic` al comparar nombres.
- Detecta identificadores duplicados y pares incompletos.
- Intenta leer cabeceras WAV sin cargar los audios completos en RAM.
- Intenta leer los JAMS como JSON e inventaría sus espacios de anotación.
- Guarda un informe fechado en JSON y muestra Python y plataforma.

Un estado `sin_incidencias_preliminares` **no certifica que las anotaciones sean
musicalmente correctas ni estén alineadas**. Esa comprobación se realizará en la fase
de datos. Si la biblioteca `wave` no admite una codificación WAV, el informe lo
señala para revisión: no debe interpretarse automáticamente como audio corrupto.

El código de salida es 0 si no se detectan incidencias preliminares y 1 si hay que
revisarlas. El informe incluye el motivo. Con `--esperados N` se puede auditar una
muestra de N grabaciones; para GuitarSet completo el valor predeterminado es 360.

## Organización prevista

| Ruta | Contenido |
|---|---|
| `data/raw/guitarset/audio_mono-mic` | WAV originales de micrófono. |
| `data/raw/guitarset/annotation` | Anotaciones originales JAMS. |
| `data/processed` | CQT y etiquetas derivadas, más adelante. |
| `splits` | Listas reproducibles de entrenamiento, validación y prueba. |
| `configs` | Configuraciones de experimentos. |
| `src` | Implementación del sistema, en etapas posteriores. |
| `scripts` | Herramientas ejecutables de apoyo. |
| `notebooks` | Cuadernos de exploración y alternativa Colab. |
| `models` | Modelos guardados. |
| `reports` | Inventarios, métricas y figuras. |
| `outputs` | Notas y tablaturas generadas. |
| `docs` | Plan, fundamentos y bitácora. |

No subas audios, características calculadas, entornos virtuales ni modelos pesados
al repositorio Git. El archivo `.gitignore` incluye exclusiones iniciales.

## Orden de lectura

1. `docs/plan_8_semanas.md`.
2. `docs/fundamentos_musicales_01.md`.
3. `docs/bitacora_y_checklists.md`.

## Fuentes

- Xi, Q., Bittner, R. M., Pauwels, J., Ye, X., & Bello, J. P. (2019).
  *GuitarSet* (versión 1.1.0) [Conjunto de datos]. Zenodo.
  https://doi.org/10.5281/zenodo.3371780
- TensorFlow. (s. f.). *Install TensorFlow with pip*.
  https://www.tensorflow.org/install/pip

La documentación de TensorFlow indica las limitaciones de GPU en Windows nativo y
su ruta basada en CUDA. Para este equipo se planifica CPU; no se presupone
aceleración con Radeon ni un tiempo de entrenamiento todavía no medido.
