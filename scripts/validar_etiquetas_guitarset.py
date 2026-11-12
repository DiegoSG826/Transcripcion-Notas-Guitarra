"""E02-T02: CLI sencilla que reutiliza la clase ValidadorEtiquetas."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from transcripcion_guitarra.data_loader import ValidadorEtiquetas, ConfiguracionEtiquetas


def main() -> None:
    parser = argparse.ArgumentParser(description='Validar anotaciones note_midi de GuitarSet (E02-T02).')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'reports/e02_t01/e02_t01_manifiesto.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/e02_t02')
    parser.add_argument('--umbral-advertencia', type=float, default=0.25)
    parser.add_argument('--umbral-ambiguedad', type=float, default=0.49)
    parser.add_argument('--tolerancia-midi', dest='umbral_ambiguedad', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    validador = ValidadorEtiquetas(ConfiguracionEtiquetas(umbral_advertencia=args.umbral_advertencia, umbral_ambiguedad=args.umbral_ambiguedad))
    resultados = validador.validar_manifiesto(args.manifest)
    rutas = validador.guardar(resultados, args.output)
    total = sum(len(r.eventos_validos) for r in resultados)
    issues = sum(len(r.incidencias) for r in resultados)
    print(f'Pistas procesadas: {len(resultados)}')
    print(f'Pistas sin incidencias: {sum(r.apto_entrenamiento for r in resultados)}')
    print(f'Pistas para revisión: {sum(not r.apto_entrenamiento for r in resultados)}')
    print(f'Eventos válidos: {total}')
    print(f'Incidencias totales: {issues}')
    print(f'Errores bloqueantes: {sum(i.severidad == "error" for r in resultados for i in r.incidencias)}')
    print(f'Advertencias: {sum(i.severidad == "advertencia" for r in resultados for i in r.incidencias)}')
    for ruta in rutas:
        print(f'Archivo: {ruta}')


if __name__ == '__main__':
    main()
