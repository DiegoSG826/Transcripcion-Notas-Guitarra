# Validación del paquete inicial

Se ejecutaron comprobaciones con archivos de ensayo creados para validar el auditor:

| Comprobación | Resultado |
|---|---|
| Pareja WAV/JAMS válida en una ruta con espacios y subcarpetas | Correcto. |
| Ejecución por línea de comandos y escritura del JSON | Correcto. |
| WAV y JAMS con nombres que no coinciden | Detectado. |
| Identificadores duplicados | Detectado. |
| JAMS con JSON inválido | Detectado. |
| Cabecera WAV ilegible | Detectado. |
| Carpetas y datos ausentes | Detectado. |
| Número esperado no positivo | Rechazado con mensaje. |

La sintaxis Python se comprobó y se verificó la integridad del ZIP entregado.
Las pruebas se ejecutaron en el entorno de preparación, basado en Linux. Las
instrucciones utilizan comandos para Windows, pero todavía no se han ejecutado en
la computadora del estudiante. El script emplea biblioteca estándar y rutas Path
para facilitar su uso en ambos sistemas.

No se ha auditado el GuitarSet descargado por el equipo: únicamente se inspeccionó
la captura adjunta. No se entrenó ninguna red ni se obtuvieron métricas de música.
No se ha comprobado todavía un entorno de TensorFlow local o Colab para este proyecto.
