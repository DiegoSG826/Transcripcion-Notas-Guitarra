"""Interfaz de consola E02-T01. Ejecutar desde la raíz del repositorio."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcripcion_guitarra.data_loader import ErrorCargaGuitarSet, GuitarSetLoader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventario WAV/JAMS GuitarSet (E02-T01)")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "guitarset")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "e02_t01")
    parser.add_argument("--tolerance-s", type=float, default=0.1)
    args = parser.parse_args()
    try:
        resultado = GuitarSetLoader(args.data_root, args.tolerance_s).inventariar()
        manifiesto, incidencias = resultado.guardar(args.output)
    except (ErrorCargaGuitarSet, ValueError, OSError) as exc:
        parser.exit(2, f"Error E02-T01: {exc}\n")
    for clave, valor in resultado.resumen.items():
        print(f"{clave}: {valor}")
    print(f"Manifiesto: {manifiesto}")
    print(f"Incidencias: {incidencias}")
    return 0 if not resultado.incidencias else 1


if __name__ == "__main__":
    sys.exit(main())
