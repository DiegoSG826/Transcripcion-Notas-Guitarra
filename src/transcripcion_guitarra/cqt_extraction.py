"""E04-T01: CQT, alineación centro-de-frame y construcción de lotes.
No considera el silencio como sinónimo de anotación ambigua.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import librosa
from .preprocessing import AudioPreparado


@dataclass(frozen=True)
class ConfiguracionCQT:
    hop_length: int = 512
    n_bins: int = 84
    bins_per_octave: int = 12
    fmin: float = 32.70319566257483  # C1
    center: bool = True
    def __post_init__(self):
        if self.hop_length < 1 or self.n_bins < 1 or self.bins_per_octave < 1 or self.fmin <= 0:
            raise ValueError("Parámetros CQT inválidos")


@dataclass(frozen=True)
class ResultadoCQT:
    cqt: np.ndarray  # float32 [T,F]
    frame_center_s: np.ndarray  # float64 [T]
    metadata: dict


class CQTExtractor:
    def __init__(self, config: ConfiguracionCQT | None = None):
        self.config = config or ConfiguracionCQT()

    def extraer(self, audio: AudioPreparado) -> ResultadoCQT:
        cfg = self.config
        x = np.asarray(audio.samples,dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Audio no finito")
        # librosa.cqt centra los frames; center=False no está disponible en su API.
        if not cfg.center:
            raise ValueError("Esta versión de librosa.cqt solo admite center=True; no aproximar tiempos")
        if x.size == 0:
            raise ValueError("CQT requiere al menos una muestra")
        if cfg.fmin * 2**((cfg.n_bins-1)/cfg.bins_per_octave) >= audio.sample_rate/2:
            raise ValueError("Rango CQT excede Nyquist; revisar sample_rate o n_bins")
        c = librosa.cqt(x, sr=audio.sample_rate, hop_length=cfg.hop_length,
                       n_bins=cfg.n_bins, bins_per_octave=cfg.bins_per_octave, fmin=cfg.fmin)
        # [F,T] -> [T,F] y compresión logarítmica reproducible
        z = np.log1p(np.abs(c)).T.astype(np.float32)
        times = librosa.frames_to_time(np.arange(z.shape[0]),sr=audio.sample_rate,hop_length=cfg.hop_length).astype(np.float64)
        if not np.isfinite(z).all():
            raise ValueError("CQT no finita")
        md = {"sample_rate":audio.sample_rate,"hop_length":cfg.hop_length,
              "n_bins":cfg.n_bins,"bins_per_octave":cfg.bins_per_octave,
              "fmin":cfg.fmin,"center":cfg.center,"scale":"log1p_magnitude"}
        return ResultadoCQT(z,times,md)


@dataclass(frozen=True)
class EtiquetasAlineadas:
    labels: np.ndarray  # int16 [T,6]
    valid_mask: np.ndarray  # bool [T] - padding temporal
    label_valid_mask: np.ndarray  # bool [T,6] - ambigüedad por cuerda


class AlineadorEtiquetas:
    @staticmethod
    def _eventos(payload):
        return payload.get("events",[]) if isinstance(payload,dict) else payload

    def alinear(self, frame_center_s, eventos) -> EtiquetasAlineadas:
        times=np.asarray(frame_center_s,dtype=np.float64)
        labels=np.zeros((len(times),6),dtype=np.int16)
        label_mask=np.ones((len(times),6),dtype=bool)
        # Se admiten registros exportados por E02-T02; los ambiguos se enmascaran.
        for ev in self._eventos(eventos):
            cuerda=int(ev["string_index"])
            if not 0 <= cuerda < 6:
                raise ValueError("Índice de cuerda fuera de rango")
            inicio=float(ev["onset_s"])
            final=float(ev.get("offset_s",inicio+float(ev.get("duration_s",0))))
            if not np.isfinite([inicio,final]).all() or final <= inicio:
                raise ValueError("Evento temporal inválido")
            activo=(times>=inicio)&(times<final)
            clase=ev.get("label_class")
            if clase is None or ev.get("estado_validacion") in ("AMBIGUO","INVALIDO","BLOQUEANTE"):
                labels[activo,cuerda]=-1
                label_mask[activo,cuerda]=False
                continue
            clase=int(clase)
            if not 1 <= clase <= 20:
                raise ValueError("Una nota activa debe tener clase 1..20")
            # Si existen solapamientos contradictorios, priorizar máscara, nunca pisar el conflicto.
            conflicto=activo & ((labels[:,cuerda]!=0)&(labels[:,cuerda]!=clase))
            labels[conflicto,cuerda]=-1
            label_mask[conflicto,cuerda]=False
            asignable=activo & label_mask[:,cuerda]
            labels[asignable,cuerda]=clase
        return EtiquetasAlineadas(labels,np.ones(len(times),dtype=bool),label_mask)


class ConstructorLotes:
    def construir(self, ejemplos: list[tuple[ResultadoCQT,EtiquetasAlineadas]]):
        if not ejemplos:
            raise ValueError("Lote vacío")
        f=ejemplos[0][0].cqt.shape[1]
        if any(c.cqt.shape[1]!=f or c.cqt.shape[0]!=e.labels.shape[0] for c,e in ejemplos):
            raise ValueError("Incompatibilidad de dimensiones")
        b=len(ejemplos); t=max(c.cqt.shape[0] for c,e in ejemplos)
        x=np.zeros((b,t,f,1),dtype=np.float32)
        y=np.full((b,t,6),-1,dtype=np.int16)
        valid=np.zeros((b,t),dtype=bool)
        label_valid=np.zeros((b,t,6),dtype=bool)
        for i,(c,e) in enumerate(ejemplos):
            n=len(c.cqt)
            x[i,:n,:,0]=c.cqt
            y[i,:n]=e.labels
            valid[i,:n]=e.valid_mask
            label_valid[i,:n]=e.label_valid_mask
        return {"x":x,"y":y,"valid_mask":valid,"label_valid_mask":label_valid}
