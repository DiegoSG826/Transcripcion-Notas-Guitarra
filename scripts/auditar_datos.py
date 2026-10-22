#!/usr/bin/env python3
r"""Auditoría preliminar de GuitarSet sin dependencias externas.

Uso desde la raíz del proyecto:
    py scripts\auditar_datos.py --preparar
    py scripts\auditar_datos.py

Comprueba archivos y pares, no la validez musical ni la alineación temporal.
No modifica audios ni anotaciones. Solo crea carpetas o informes de diagnóstico.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# Estas carpetas también se usarán en las siguientes fases del proyecto.
CARPETAS = (
    "data/raw/guitarset/audio_mono-mic",
    "data/raw/guitarset/annotation",
    "data/processed", "splits", "configs", "src", "notebooks",
    "models", "reports", "outputs",
)


def archivos_de(carpeta: Path, extension: str) -> list[Path]:
    """Busca recursivamente; ignora metadatos auxiliares de ZIP creados en macOS."""
    if not carpeta.is_dir():
        return []
    return sorted(
        p for p in carpeta.rglob("*")
        if p.is_file() and p.suffix.lower() == extension
        and "__MACOSX" not in p.parts and not p.name.startswith("._")
    )


def agrupar_por_id(archivos: list[Path], audio: bool) -> dict[str, list[Path]]:
    """El WAV termina en _mic; su JAMS normalmente comparte el resto del nombre."""
    grupos: dict[str, list[Path]] = defaultdict(list)
    for ruta in archivos:
        identificador = ruta.stem
        if audio and identificador.endswith("_mic"):
            identificador = identificador[:-4]
        grupos[identificador].append(ruta)
    return dict(grupos)


def auditar(raiz: Path, esperados: int) -> dict:
    """Reúne incidencias y metadatos sin cargar las muestras de audio en memoria."""
    carpeta_audio = raiz / CARPETAS[0]
    carpeta_anotaciones = raiz / CARPETAS[1]
    audios = archivos_de(carpeta_audio, ".wav")
    anotaciones = archivos_de(carpeta_anotaciones, ".jams")
    ids_audio = agrupar_por_id(audios, audio=True)
    ids_anotaciones = agrupar_por_id(anotaciones, audio=False)
    incidencias: list[str] = []

    for nombre, carpeta, lista in (
        ("WAV", carpeta_audio, audios), ("JAMS", carpeta_anotaciones, anotaciones)
    ):
        if not carpeta.is_dir():
            incidencias.append(f"No existe la carpeta de {nombre}: {carpeta}")
        if len(lista) != esperados:
            incidencias.append(f"Se esperaban {esperados} {nombre}; se encontraron {len(lista)}.")

    duplicados_audio = {k: [str(p) for p in v] for k, v in ids_audio.items() if len(v) > 1}
    duplicados_jams = {k: [str(p) for p in v] for k, v in ids_anotaciones.items() if len(v) > 1}
    sin_anotacion = sorted(ids_audio.keys() - ids_anotaciones.keys())
    sin_audio = sorted(ids_anotaciones.keys() - ids_audio.keys())
    if duplicados_audio or duplicados_jams:
        incidencias.append("Hay identificadores duplicados; no se debe elegir una copia arbitrariamente.")
    if sin_anotacion or sin_audio:
        incidencias.append("Hay archivos sin pareja audio–anotación.")

    cabeceras: list[dict] = []
    errores_wav: list[dict] = []
    canales = Counter()
    frecuencias = Counter()
    duracion = 0.0
    for ruta in audios:
        if not ruta.stem.endswith("_mic"):
            incidencias.append(f"Nombre WAV sin sufijo _mic; comprobar su procedencia: {ruta.name}")
        try:
            # wave lee la cabecera. No se llama a readframes: el consumo de RAM es bajo.
            with wave.open(str(ruta), "rb") as wav:
                n_canales = wav.getnchannels()
                fs = wav.getframerate()
                n_muestras = wav.getnframes()
                if fs <= 0 or n_canales <= 0 or n_muestras <= 0:
                    raise ValueError("Cabecera con frecuencia, canales o longitud no positivos.")
                segundos = n_muestras / fs
                cabeceras.append({"archivo": str(ruta), "canales": n_canales,
                                  "muestreo_hz": fs, "duracion_s": segundos,
                                  "bytes_por_muestra": wav.getsampwidth()})
                canales[n_canales] += 1
                frecuencias[fs] += 1
                duracion += segundos
        except (OSError, EOFError, wave.Error, ValueError) as error:
            errores_wav.append({"archivo": str(ruta), "detalle": str(error)})
    if errores_wav:
        incidencias.append("Hay cabeceras WAV ilegibles o codificaciones no admitidas por wave; revisar.")
    if any(n != 1 for n in canales):
        incidencias.append("Hay WAV con más de un canal; comprobar que se extrajo audio_mono-mic.")

    espacios = Counter()
    errores_jams: list[dict] = []
    for ruta in anotaciones:
        try:
            with ruta.open("r", encoding="utf-8-sig") as archivo:
                contenido = json.load(archivo)
            # Es una comprobación estructural mínima, no una validación del esquema JAMS.
            if not isinstance(contenido, dict):
                raise ValueError("La raíz JSON no es un objeto.")
            lista = contenido.get("annotations")
            if not isinstance(lista, list) or not lista:
                raise ValueError("annotations no es una lista no vacía.")
            for anotacion in lista:
                if not isinstance(anotacion, dict):
                    raise ValueError("Una anotación no es un objeto.")
                espacio = anotacion.get("namespace")
                if not isinstance(espacio, str) or not espacio:
                    raise ValueError("Una anotación no tiene namespace válido.")
                espacios[espacio] += 1
        except (OSError, UnicodeError, ValueError) as error:
            errores_jams.append({"archivo": str(ruta), "detalle": str(error)})
    if errores_jams:
        incidencias.append("Hay JAMS que requieren revisión de lectura o estructura mínima.")

    pares_unicos = sorted(
        clave for clave in ids_audio.keys() & ids_anotaciones.keys()
        if len(ids_audio[clave]) == 1 and len(ids_anotaciones[clave]) == 1
    )
    return {
        "estado": "requiere_revision" if incidencias else "sin_incidencias_preliminares",
        "alcance": "Inventario, emparejamiento, JSON y cabeceras; no valida música ni alineación.",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "entorno": {"python": platform.python_version(), "plataforma": platform.platform(),
                    "procesadores_logicos": os.cpu_count()},
        "raiz": str(raiz), "esperados_por_tipo": esperados,
        "wav_encontrados": len(audios), "jams_encontrados": len(anotaciones),
        "pares_unicos_por_nombre": len(pares_unicos),
        "duracion_total_cabeceras_legibles_s": duracion,
        "distribucion_canales": dict(canales), "distribucion_muestreo_hz": dict(frecuencias),
        "inventario_namespaces_jams": dict(espacios),
        "sin_anotacion": sin_anotacion, "sin_audio": sin_audio,
        "duplicados_audio": duplicados_audio, "duplicados_jams": duplicados_jams,
        "errores_wav": errores_wav, "errores_jams": errores_jams,
        "cabeceras_wav": cabeceras, "incidencias": incidencias,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Raíz del proyecto; por defecto, la carpeta que contiene scripts.")
    parser.add_argument("--esperados", type=int, default=360,
                        help="Cantidad esperada de WAV y JAMS; usa 360 para GuitarSet completo.")
    parser.add_argument("--preparar", action="store_true",
                        help="Solo crea las carpetas; no descarga, mueve ni elimina archivos.")
    argumentos = parser.parse_args()
    if argumentos.esperados <= 0:
        parser.error("--esperados debe ser un entero positivo.")
    raiz = argumentos.raiz.expanduser().resolve()
    try:
        if argumentos.preparar:
            for relativa in CARPETAS:
                (raiz / relativa).mkdir(parents=True, exist_ok=True)
            print(f"Carpetas preparadas en: {raiz}")
            return 0

        informe = auditar(raiz, argumentos.esperados)
        carpeta_informes = raiz / "reports"
        carpeta_informes.mkdir(parents=True, exist_ok=True)
        sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        destino = carpeta_informes / f"auditoria_datos_{sello}.json"
        # El nombre fechado evita reemplazar informes de ejecuciones anteriores.
        with destino.open("x", encoding="utf-8") as archivo:
            json.dump(informe, archivo, ensure_ascii=False, indent=2)
        print(f"Estado: {informe['estado']}")
        print(f"WAV: {informe['wav_encontrados']} | JAMS: {informe['jams_encontrados']} | "
              f"Pares únicos: {informe['pares_unicos_por_nombre']}")
        print(f"Python: {informe['entorno']['python']}")
        for incidencia in informe["incidencias"]:
            print(f"- {incidencia}")
        print(f"Informe: {destino}")
        print("Esta revisión no certifica la corrección musical ni la alineación temporal.")
        return 1 if informe["incidencias"] else 0
    except OSError as error:
        print(f"No se pudo completar la operación de archivos: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
