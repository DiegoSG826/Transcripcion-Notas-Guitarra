"""E05: Arquitectura CRNN orientada a objetos para transcripción de guitarra.

Entrada esperada: [B, T, F, 1] (float32, magnitud logarítmica CQT).
Salida: [B, T, 6, 21] (float32, probabilidades softmax por cuerda).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tensorflow as tf
from keras import layers, models

from .domain import NUM_CLASES_POR_CUERDA, NUM_CUERDAS


@dataclass(frozen=True)
class ConfiguracionCRNN:
    n_bins: int = 84
    conv_filters: tuple[int, ...] = (32, 64, 128)
    lstm_units: int = 128
    dropout_rate: float = 0.3
    num_cuerdas: int = NUM_CUERDAS
    num_clases: int = NUM_CLASES_POR_CUERDA

    def __post_init__(self) -> None:
        if self.n_bins < 8:
            raise ValueError("n_bins debe ser al menos 8 para soportar los bloques de pooling.")
        if not self.conv_filters or any(f <= 0 for f in self.conv_filters):
            raise ValueError("conv_filters debe contener enteros positivos.")
        if self.lstm_units <= 0:
            raise ValueError("lstm_units debe ser positivo.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate debe estar en [0, 1).")


class CRNNTranscriptor:
    """Modelo CRNN que preserva el eje temporal T y clasifica 6 cuerdas en paralelo."""

    def __init__(self, config: ConfiguracionCRNN | None = None) -> None:
        self.config = config or ConfiguracionCRNN()
        self.modelo: tf.keras.Model = self._construir_modelo()

    def _construir_modelo(self) -> tf.keras.Model:
        cfg = self.config
        entrada = layers.Input(shape=(None, cfg.n_bins, 1), name="cqt_input")
        x = entrada

        # Bloques convolucionales: pooling solo en frecuencia (1, 2) para no alterar T
        for idx, filtros in enumerate(cfg.conv_filters, start=1):
            x = layers.Conv2D(
                filters=filtros,
                kernel_size=(3, 3),
                padding="same",
                use_bias=False,
                name=f"conv_{idx}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_{idx}")(x)
            x = layers.ReLU(name=f"relu_{idx}")(x)
            x = layers.MaxPooling2D(pool_size=(1, 2), name=f"pool_freq_{idx}")(x)
            x = layers.Dropout(cfg.dropout_rate * 0.5, name=f"drop_conv_{idx}")(x)

        # Aplanar frecuencia y canales conservando T dinámico: [B, T, F_red * C]
        f_reducida = x.shape[2]
        canales = x.shape[3]
        x = layers.Reshape((-1, f_reducida * canales), name="reshape_temporal")(x)

        # Bloque recurrente bidireccional
        x = layers.Bidirectional(
            layers.LSTM(cfg.lstm_units, return_sequences=True),
            name="bilstm_1",
        )(x)
        x = layers.Dropout(cfg.dropout_rate, name="drop_lstm_1")(x)

        x = layers.Bidirectional(
            layers.LSTM(cfg.lstm_units, return_sequences=True),
            name="bilstm_2",    
        )(x)
        x = layers.Dropout(cfg.dropout_rate, name="drop_lstm_2")(x)

        # Cabeza multitarea: 6 cuerdas x 21 clases
        x = layers.Dense(cfg.num_cuerdas * cfg.num_clases, name="logits_planos")(x)
        x = layers.Reshape((-1, cfg.num_cuerdas, cfg.num_clases), name="logits_cuerdas")(x)
        salida = layers.Softmax(axis=-1, name="probabilidades_cuerdas")(x)

        return models.Model(inputs=entrada, outputs=salida, name="CRNNTranscriptor")

    def predecir_probabilidades(self, x: np.ndarray | tf.Tensor, training: bool = False) -> tf.Tensor:
        """Devuelve probabilidades [B, T, 6, 21]."""
        return self.modelo(x, training=training)

    def predecir_clases(self, x: np.ndarray | tf.Tensor) -> np.ndarray:
        """Devuelve clases enteras [B, T, 6] en 0..20 listas para contar_aciertos_enmascarados."""
        probs = self.predecir_probabilidades(x, training=False)
        return tf.argmax(probs, axis=-1, output_type=tf.int32).numpy().astype(np.int16)

    def resumen(self) -> None:
        self.modelo.summary()

    def guardar(self, ruta: str | Path) -> Path:
        destino = Path(ruta)
        destino.parent.mkdir(parents=True, exist_ok=True)
        self.modelo.save(destino)
        return destino

    @classmethod
    def cargar(cls, ruta: str | Path, config: ConfiguracionCRNN | None = None) -> "CRNNTranscriptor":
        instancia = cls.__new__(cls)
        instancia.config = config or ConfiguracionCRNN()
        instancia.modelo = models.load_model(ruta)
        return instancia