"""Entidades y reglas de dominio compartidas por los módulos del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

NUM_CUERDAS = 6
NUM_CLASES_POR_CUERDA = 21
CLASE_SILENCIO = 0
TRASTE_MINIMO = 0
TRASTE_MAXIMO = 19
MIDI_CUERDAS_ABIERTAS = (40, 45, 50, 55, 59, 64)


@dataclass(frozen=True)
class ConfiguracionAudio:
    """Parámetros que identifican cómo se leyó y representó un WAV."""

    sample_rate: int
    hop_length: int
    center: bool


@dataclass(frozen=True)
class MetadatosFrame:
    """Tiempo y validez de los frames de una representación temporal."""

    frame_center_s: Sequence[float]
    valid_mask: Sequence[bool]

    def __post_init__(self) -> None:
        if len(self.frame_center_s) != len(self.valid_mask):
            raise ValueError("frame_center_s y valid_mask deben tener la misma longitud.")

    @property
    def cantidad_frames(self) -> int:
        return len(self.frame_center_s)


@dataclass(frozen=True)
class EventoNota:
    """Evento musical normalizado para una cuerda de guitarra."""

    track_id: str
    string_index: int
    onset_s: float
    offset_s: float
    fret: int | None

    def __post_init__(self) -> None:
        if not 0 <= self.string_index < NUM_CUERDAS:
            raise ValueError("string_index debe estar entre 0 y 5.")
        if self.offset_s <= self.onset_s:
            raise ValueError("Todo evento debe tener duración positiva.")
        if self.fret is not None and not TRASTE_MINIMO <= self.fret <= TRASTE_MAXIMO:
            raise ValueError("El traste debe estar entre 0 y 19 o ser None.")

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s

    @property
    def label_class(self) -> int:
        return CLASE_SILENCIO if self.fret is None else self.fret + 1


@dataclass(frozen=True)
class ResultadoInferencia:
    """Resultado de E08 que consumen la CLI y la demo web local."""

    source_wav: Path
    tablatura_ascii: str
    events: Sequence[EventoNota]
    config_path: Path
    checkpoint_path: Path
