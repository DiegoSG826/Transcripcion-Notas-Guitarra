
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcripcion_guitarra.preprocessing import (
    ConfiguracionPreprocesamiento,
    AudioPreprocessor,
    SegmentadorAudio,
)

config = ConfiguracionPreprocesamiento(
    target_sample_rate=None,
    peak_normalize=True,
    peak_target=0.95,
    window_s=2.0,
    hop_s=1.0,
    pad_last=True,
)

ruta = ROOT / (
    "data/raw/guitarset/audio_mono-mic/"
    "00_BN1-129-Eb_comp_mic.wav"
)

audio = AudioPreprocessor(config).procesar(ruta)
ventanas = SegmentadorAudio(config).segmentar(audio)

sr = audio.sample_rate
ancho = round(config.window_s * sr)
salto = round(config.hop_s * sr)

for i, ventana in enumerate(ventanas):
    inicio_muestra = i * salto
    fin_muestra = inicio_muestra + ventana.valid_samples

    assert len(ventana.samples) == ancho
    assert np.isclose(ventana.start_s, inicio_muestra / sr)
    assert np.isclose(ventana.end_s, fin_muestra / sr)
    assert np.all(np.isfinite(ventana.samples))

    original = audio.samples[inicio_muestra:fin_muestra]
    assert np.array_equal(
        ventana.samples[:ventana.valid_samples],
        original
    )

    if ventana.padded:
        assert np.all(ventana.samples[ventana.valid_samples:] == 0)

    print(
        f"Ventana {i+1}: "
        f"{ventana.start_s:.3f}-{ventana.end_s:.3f}s | "
        f"validas={ventana.valid_samples} | "
        f"relleno={ventana.padded}"
    )

print(f"\nTrazabilidad verificada: {len(ventanas)} ventanas.")
