"""Paquete de dominio para la transcripción de guitarra."""

from .domain import (
    CLASE_SILENCIO,
    NUM_CLASES_POR_CUERDA,
    NUM_CUERDAS,
    EventoNota,
    MetadatosFrame,
    ResultadoInferencia,
)

__all__ = [
    "CLASE_SILENCIO",
    "NUM_CLASES_POR_CUERDA",
    "NUM_CUERDAS",
    "EventoNota",
    "MetadatosFrame",
    "ResultadoInferencia",
]
