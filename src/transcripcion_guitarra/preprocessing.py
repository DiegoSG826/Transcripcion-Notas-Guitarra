"""E03-T01: lectura mono, normalización opcional y ventanas trazables."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ConfiguracionPreprocesamiento:
    target_sample_rate: int | None = None
    peak_normalize: bool = True
    peak_target: float = 0.95
    window_s: float = 2.0
    hop_s: float = 1.0
    pad_last: bool = True
    def __post_init__(self):
        if self.target_sample_rate is not None and self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate debe ser positivo")
        if not (0 < self.peak_target <= 1):
            raise ValueError("peak_target debe estar en (0,1]")
        if self.window_s <= 0 or self.hop_s <= 0:
            raise ValueError("window_s y hop_s deben ser positivos")


@dataclass(frozen=True)
class AudioPreparado:
    samples: np.ndarray
    sample_rate: int
    duration_original_s: float
    transformaciones: tuple[str, ...]
    source: str
    @property
    def duration_s(self):
        return len(self.samples)/self.sample_rate


@dataclass(frozen=True)
class SegmentoAudio:
    samples: np.ndarray
    start_s: float
    end_s: float
    valid_samples: int
    padded: bool
    sample_rate: int


class AudioPreprocessor:
    def __init__(self, config: ConfiguracionPreprocesamiento | None = None):
        self.config = config or ConfiguracionPreprocesamiento()

    def procesar(self, wav_path: str | Path) -> AudioPreparado:
        wav_path = Path(wav_path)
        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        if sr <= 0:
            raise ValueError("Frecuencia de muestreo inválida")
        dur = len(data)/sr
        if not np.all(np.isfinite(data)):
            raise ValueError(f"Audio con NaN o infinito: {wav_path}")
        ops = []
        mono = data.mean(axis=1, dtype=np.float64).astype(np.float32)
        ops.append(f"mono_promedio_{data.shape[1]}_canales")
        cfg = self.config
        if cfg.target_sample_rate is not None and cfg.target_sample_rate != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, cfg.target_sample_rate)
            mono = resample_poly(mono, cfg.target_sample_rate//g, sr//g).astype(np.float32)
            ops.append(f"resample_poly_{sr}_a_{cfg.target_sample_rate}")
            sr = cfg.target_sample_rate
        if cfg.peak_normalize and len(mono):
            peak = float(np.max(np.abs(mono)))
            if peak > 0:
                mono = (mono*(cfg.peak_target/peak)).astype(np.float32)
                ops.append(f"normalizacion_pico_{cfg.peak_target}")
            else:
                ops.append("silencio_sin_division")
        if not np.all(np.isfinite(mono)):
            raise ValueError("Preprocesamiento produjo NaN o infinito")
        return AudioPreparado(mono, sr, dur, tuple(ops), str(wav_path))


class SegmentadorAudio:
    def __init__(self, config: ConfiguracionPreprocesamiento | None = None):
        self.config = config or ConfiguracionPreprocesamiento()

    def segmentar(self, audio: AudioPreparado) -> list[SegmentoAudio]:
        sr = audio.sample_rate
        width = max(1, round(self.config.window_s*sr))
        hop = max(1, round(self.config.hop_s*sr))
        n = len(audio.samples)
        if n == 0:
            return [SegmentoAudio(np.zeros(width,dtype=np.float32),0.,0.,0,True,sr)] if self.config.pad_last else []
        salida = []
        for start in range(0,n,hop):
            valid = min(width,n-start)
            if valid < width and not self.config.pad_last:
                break
            window = audio.samples[start:start+valid]
            padded = valid < width
            if padded:
                window = np.pad(window,(0,width-valid)).astype(np.float32)
            salida.append(SegmentoAudio(window,start/sr,(start+valid)/sr,valid,padded,sr))
            if start+width >= n:
                break
        return salida
