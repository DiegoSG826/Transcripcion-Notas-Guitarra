"""Demo CQT con catálogo completo de E02 y máscaras de supervisión."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from transcripcion_guitarra.preprocessing import AudioPreprocessor, ConfiguracionPreprocesamiento
from transcripcion_guitarra.cqt_extraction import CQTExtractor, ConfiguracionCQT, EtiquetasAlineadas
from transcripcion_guitarra.label_policy import CatalogoEtiquetas
from transcripcion_guitarra.supervision import mascara_supervision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/etiquetas.json')
    parser.add_argument('--events', type=Path, help='Compatibilidad: debe coincidir con la fuente del catálogo')
    parser.add_argument('--sin-etiquetas', action='store_true', help='WAV externo: toda supervisión se desactiva')
    parser.add_argument('--output', type=Path, default=Path('reports/e04_demo/cqt_demo.npz'))
    args = parser.parse_args()
    if args.sin_etiquetas and args.events:
        parser.error('--sin-etiquetas no admite --events')
    catalogo = None if args.sin_etiquetas else CatalogoEtiquetas.cargar(ROOT, args.config)
    track = args.wav.stem.removesuffix('_mic')
    if catalogo is not None:
        catalogo.eventos_pista(track)
        fuente = ROOT / catalogo.procedencia['fuentes']['eventos']['ruta']
        if args.events and args.events.resolve() != fuente.resolve():
            parser.error('--events difiere de la fuente configurada; use un --config coherente')
    audio = AudioPreprocessor(ConfiguracionPreprocesamiento(peak_normalize=False)).procesar(args.wav)
    cqt = CQTExtractor(ConfiguracionCQT()).extraer(audio)
    if catalogo is None:
        valid = cqt.frame_center_s < audio.duration_s
        alineadas = EtiquetasAlineadas(np.full((len(valid), 6), -1, dtype=np.int16),
                                       valid, np.zeros((len(valid), 6), dtype=bool))
    else:
        if abs(audio.duration_s - catalogo.pistas[track]['duration_s']) > catalogo.config.tolerancia_tiempo_s:
            raise ValueError('La duración del WAV no corresponde al manifiesto')
        alineadas = catalogo.alinear_pista(track, cqt.frame_center_s, fin_valido_s=audio.duration_s)
    supervision = mascara_supervision(alineadas.labels, alineadas.valid_mask, alineadas.label_valid_mask)
    metadata = dict(cqt.metadata, track_id=track, tipo_etiquetas='sin_supervision' if catalogo is None else 'catalogo_e02')
    if catalogo is not None:
        metadata['etiquetas'] = catalogo.procedencia
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cqt=cqt.cqt, frame_center_s=cqt.frame_center_s,
                        labels=alineadas.labels, valid_mask=alineadas.valid_mask,
                        label_valid_mask=alineadas.label_valid_mask, supervision_mask=supervision,
                        metadata=json.dumps(metadata))
    print('CQT:', cqt.cqt.shape, 'Etiquetas:', alineadas.labels.shape)
    print('Posiciones-cuerda supervisadas:', int(supervision.sum()))
    print('Archivo:', out)


if __name__ == '__main__':
    main()
