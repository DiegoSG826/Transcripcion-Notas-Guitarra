#!/usr/bin/env python3
"""Descarga y organiza los archivos de GuitarSet desde Zenodo.

Uso desde la raíz del repositorio:
    python scripts/preparar_datos.py

Descarga annotation.zip y audio_mono-mic.zip, comprueba sus MD5 publicados
por Zenodo y organiza los archivos en:
    data/raw/guitarset/annotation/
    data/raw/guitarset/audio_mono-mic/

Solo usa la biblioteca estándar de Python.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "guitarset"
DOWNLOAD_ROOT = ROOT / "data" / "downloads"

FILES = [
    {
        "name": "annotation.zip",
        "url": "https://zenodo.org/records/3371780/files/annotation.zip?download=1",
        "md5": "b39b78e63d3446f2e54ddb7a54df9b10",
        "target": DATA_ROOT / "annotation",
        "suffix": ".jams",
    },
    {
        "name": "audio_mono-mic.zip",
        "url": "https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1",
        "md5": "275966d6610ac34999b58426beb119c3",
        "target": DATA_ROOT / "audio_mono-mic",
        "suffix": ".wav",
    },
]


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """Descarga a un archivo temporal y lo renombra al terminar."""
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "Transcripcion-Notas-Guitarra/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as out:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    percent = downloaded * 100 / total_bytes
                    print(f"\r  Descargado: {percent:5.1f}%", end="", flush=True)
        print()
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def is_already_extracted(folder: Path, suffix: str) -> bool:
    return folder.exists() and any(folder.rglob(f"*{suffix}"))


def safe_output_path(root: Path, member_name: str) -> Path:
    """Evita que un ZIP escriba fuera del directorio de destino."""
    normalized = member_name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Ruta insegura dentro del ZIP: {member_name}")
    return root.joinpath(*relative.parts)


def extract_matching_files(zip_path: Path, target: Path, suffix: str) -> int:
    """Extrae únicamente archivos con la extensión esperada."""
    target.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist()
                   if not m.is_dir() and m.filename.lower().endswith(suffix)]
        if not members:
            raise RuntimeError(
                f"No se encontraron archivos {suffix} dentro de {zip_path.name}."
            )
        for member in members:
            output = safe_output_path(target, Path(member.filename).name)
            # Si el ZIP contiene subcarpetas, conservar solo el nombre del archivo:
            # GuitarSet usa nombres únicos por pista en estos archivos.
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, output.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            extracted += 1
    return extracted


def main() -> int:
    if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
        print(
            f"AVISO: el proyecto recomienda Python 3.12; estás usando "
            f"{sys.version.split()[0]}.",
            file=sys.stderr,
        )

    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for item in FILES:
        archive_path = DOWNLOAD_ROOT / item["name"]
        target = item["target"]
        suffix = item["suffix"]

        if is_already_extracted(target, suffix):
            count = sum(1 for _ in target.rglob(f"*{suffix}"))
            print(f"[OMITIDO] Ya existen archivos en {target} ({count} {suffix}).")
            print("          Si necesitas reinstalarlos, revisa la carpeta manualmente.")
            continue

        print(f"\n==> Preparando {item['name']}")
        if not archive_path.exists():
            print(f"Descargando desde Zenodo: {item['url']}")
            download_file(item["url"], archive_path)
        else:
            print(f"Archivo ZIP existente: {archive_path}")

        actual_md5 = md5_file(archive_path)
        if actual_md5.lower() != item["md5"]:
            raise RuntimeError(
                f"MD5 incorrecto para {item['name']}.\n"
                f"Esperado: {item['md5']}\nObtenido: {actual_md5}\n"
                "El ZIP se conserva para inspección; elimínalo y vuelve a ejecutar."
            )
        print("MD5 verificado correctamente.")

        count = extract_matching_files(archive_path, target, suffix)
        print(f"Extraídos {count} archivos {suffix} en: {target}")

    wav_count = sum(1 for _ in (DATA_ROOT / "audio_mono-mic").rglob("*.wav"))
    jams_count = sum(1 for _ in (DATA_ROOT / "annotation").rglob("*.jams"))
    print("\n==> Resumen")
    print(f"WAV:  {wav_count}")
    print(f"JAMS: {jams_count}")
    print(f"Directorio: {DATA_ROOT}")

    if wav_count == 360 and jams_count == 360:
        print("Dataset listo para ejecutar la auditoría.")
    else:
        print(
            "AVISO: se esperaba encontrar 360 WAV y 360 JAMS. "
            "Revisa los archivos y ejecuta scripts/auditar_datos.py."
        )
    print("Nota: MD5 verifica la integridad del archivo descargado, no su contenido musical.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
