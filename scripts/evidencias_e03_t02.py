"""E03-T02: ejecutar casos sintéticos y guardar resumen reproducible JSON + figuras."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from transcripcion_guitarra.preprocessing import (  # noqa: E402
    AudioPreprocessor, SegmentadorAudio, ConfiguracionPreprocesamiento
)

def analizar(nombre: str, datos: np.ndarray, directorio: Path, config: ConfiguracionPreprocesamiento, sr: int) -> dict:
    wav = directorio / f'{nombre}.wav'
    sf.write(wav, datos.astype(np.float32), sr, subtype='FLOAT')
    audio = AudioPreprocessor(config).procesar(wav)
    ventanas = SegmentadorAudio(config).segmentar(audio)
    assert np.isfinite(audio.samples).all(), nombre
    assert all(np.isfinite(s.samples).all() for s in ventanas), nombre
    assert all(len(s.samples) == round(config.window_s * sr) for s in ventanas), nombre
    assert all(np.all(s.samples[s.valid_samples:] == 0) for s in ventanas), nombre
    assert all(np.allclose(s.samples[:s.valid_samples], audio.samples[round(s.start_s*sr):round(s.start_s*sr)+s.valid_samples],atol=1e-7) for s in ventanas), nombre
    return {
        'caso': nombre,
        'duracion_original_s': audio.duration_original_s,
        'frecuencia_muestreo': audio.sample_rate,
        'pico_procesado': float(np.max(np.abs(audio.samples))) if len(audio.samples) else 0.,
        'transformaciones': list(audio.transformaciones),
        'cantidad_ventanas': len(ventanas),
        'ventanas': [dict(inicio_s=s.start_s,fin_real_s=s.end_s,muestras_validas=s.valid_samples,
                          muestras_totales=len(s.samples),relleno=s.padded) for s in ventanas]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/e03_t02')
    args=parser.parse_args()
    out=ROOT / args.output
    out.mkdir(parents=True,exist_ok=True)
    sr=8000
    cfg=ConfiguracionPreprocesamiento(window_s=2,hop_s=1,peak_normalize=True,peak_target=.95,pad_last=True)
    with tempfile.TemporaryDirectory() as td:
        work=Path(td)
        t=np.arange(sr//2)/sr
        casos=[
            ('silencio_3s',np.zeros(sr*3,dtype=np.float32)),
            ('audio_corto_0_5s',(0.25*np.sin(2*np.pi*220*t)).astype(np.float32)),
            ('ultima_incompleta_2_3s',np.linspace(-.3,.4,int(sr*2.3),dtype=np.float32))
        ]
        resultados=[analizar(n,x,work,cfg,sr) for n,x in casos]
    info={'tarea':'E03-T02', 'tipo_datos':'sinteticos',
          'configuracion': {'sr_hz':sr,'window_s':2.0,'hop_s':1.0,'peak_target':.95,'pad_last':True},
          'casos':resultados}
    output=out/'e03_t02_resultados.json'
    output.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    for r in resultados:
        print(f"{r['caso']}: {r['duracion_original_s']:.2f} s, {r['cantidad_ventanas']} ventana(s)")
        for v in r['ventanas']:
            print(f"  {v['inicio_s']:.2f}–{v['fin_real_s']:.2f} s, válidas={v['muestras_validas']}, relleno={v['relleno']}")
    print('Evidencia:',output)

if __name__=='__main__':
    main()
