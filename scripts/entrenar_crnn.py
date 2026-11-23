"""Script CLI para ejecutar el entrenamiento E06 de CRNNTranscriptor."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcripcion_guitarra.cqt_extraction import ConfiguracionCQT
from transcripcion_guitarra.model_crnn import ConfiguracionCRNN, CRNNTranscriptor
from transcripcion_guitarra.preprocessing import ConfiguracionPreprocesamiento
from transcripcion_guitarra.train import (
    ConfiguracionEntrenamiento,
    EntrenadorCRNN,
    ProveedorLotesGuitarSet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento CRNN (Etapa E06)")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT.parent / "data" / "GuitarSet",
        help="Ruta raíz que contiene audio_mono-mic/ y annotation/",
    )
    parser.add_argument(
        "--config-e02",
        type=str,
        default="configs/config_e02.json",
        help="Ruta relativa dentro del repo al JSON de configuración E02",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-pistas",
        type=int,
        default=None,
        help="Limita pistas por época (ej. 8 para prueba rápida en CPU)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prep_cfg = ConfiguracionPreprocesamiento(target_sample_rate=22050, window_s=2.0, hop_s=1.0)
    cqt_cfg = ConfiguracionCQT(hop_length=512, n_bins=84, bins_per_octave=12)
    modelo_cfg = ConfiguracionCRNN(n_bins=cqt_cfg.n_bins)
    train_cfg = ConfiguracionEntrenamiento(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_pistas_por_epoca=args.max_pistas,
    )

    proveedor = ProveedorLotesGuitarSet(
        data_root=args.data_root,
        config_json_path=args.config_e02,
        repo_root=ROOT,
        prep_config=prep_cfg,
        cqt_config=cqt_cfg,
    )
    modelo = CRNNTranscriptor(modelo_cfg)
    entrenador = EntrenadorCRNN(modelo, train_cfg)

    print(
        f"Iniciando E06 | Pistas train={len(proveedor.pistas_de('train'))}, "
        f"val={len(proveedor.pistas_de('validation'))}"
    )
    reporte = entrenador.entrenar(proveedor, proveedor)
    print(f"Entrenamiento finalizado. Mejor pérdida: {reporte['mejor_loss']}")


if __name__ == "__main__":
    main()