"""Máscara común para E04, pérdida de E05/E06 y métricas de E07.

Las exclusiones se aplican por posición temporal y cuerda, antes de calcular
la pérdida. Un -1 nunca se convierte en un objetivo de silencio.
"""
from __future__ import annotations

import numpy as np

from .domain import NUM_CLASES_POR_CUERDA, NUM_CUERDAS


def mascara_supervision(labels, valid_mask, label_valid_mask) -> np.ndarray:
    """Combina máscaras [T] / [T,6] o [B,T] / [B,T,6].

    Exige etiquetas enteras -1..20 y rechaza un -1 sin su máscara de exclusión.
    Se permite una etiqueta conocida bajo máscara falsa (por ejemplo padding).
    """
    y = np.asarray(labels)
    temporal = np.asarray(valid_mask)
    por_cuerda = np.asarray(label_valid_mask)
    if y.ndim not in (2, 3) or y.shape[-1] != NUM_CUERDAS:
        raise ValueError("labels debe tener forma [T,6] o [B,T,6]")
    if temporal.shape != y.shape[:-1] or por_cuerda.shape != y.shape:
        raise ValueError("Dimensiones de máscaras incompatibles con labels")
    if temporal.dtype != np.bool_ or por_cuerda.dtype != np.bool_:
        raise ValueError("Las máscaras deben ser booleanas")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Las etiquetas deben ser enteras")
    if np.any((y < -1) | (y >= NUM_CLASES_POR_CUERDA)):
        raise ValueError("Etiquetas fuera de -1..20")
    mask = temporal[..., None] & por_cuerda
    if np.any(mask & (y < 0)):
        raise ValueError("Etiqueta -1 sin máscara de exclusión")
    return mask


def contar_aciertos_enmascarados(labels, predicciones, valid_mask, label_valid_mask):
    """Contadores acumulables entre lotes; los excluidos no cuentan como aciertos.

    Recibe clases predichas, no probabilidades. Una cuerda sin supervisión tiene
    exactitud None, para no presentarla como un resultado perfecto o nulo.
    """
    y = np.asarray(labels)
    pred = np.asarray(predicciones)
    mask = mascara_supervision(y, valid_mask, label_valid_mask)
    if pred.shape != y.shape or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("predicciones debe contener clases enteras con forma de labels")
    if np.any((pred[mask] < 0) | (pred[mask] >= NUM_CLASES_POR_CUERDA)):
        raise ValueError("Predicción supervisada fuera de 0..20")
    ejes = tuple(range(y.ndim - 1))
    evaluados = mask.sum(axis=ejes)
    aciertos = ((pred == y) & mask).sum(axis=ejes)
    return {
        "evaluados_por_cuerda": evaluados.tolist(),
        "aciertos_por_cuerda": aciertos.tolist(),
        "excluidos_por_cuerda": (~mask).sum(axis=ejes).tolist(),
        "exactitud_por_cuerda": [int(a) / int(n) if n else None for a, n in zip(aciertos, evaluados)],
        "evaluados": int(evaluados.sum()),
        "aciertos": int(aciertos.sum()),
    }


def entropia_cruzada_enmascarada(labels, predicciones, valid_mask, label_valid_mask,
                                *, from_logits=False):
    """Pérdida TensorFlow diferenciable, media por posición-cuerda supervisada.

    Acepta [T,6,21] o [B,T,6,21]. Selecciona objetivos y predicciones ANTES
    de la entropía cruzada, evitando usar -1 o multiplicar pérdidas inválidas
    por cero. Un lote sin supervisión devuelve cero con gradientes cero;
    el entrenador debe omitir su paso de optimización.
    """
    import tensorflow as tf

    y = tf.convert_to_tensor(labels)
    pred = tf.convert_to_tensor(predicciones)
    temporal = tf.convert_to_tensor(valid_mask)
    por_cuerda = tf.convert_to_tensor(label_valid_mask)
    if not y.dtype.is_integer or not pred.dtype.is_floating:
        raise ValueError("Se requieren etiquetas enteras y predicciones flotantes")
    if temporal.dtype != tf.bool or por_cuerda.dtype != tf.bool:
        raise ValueError("Las máscaras deben ser booleanas")
    tf.debugging.assert_rank_in(y, [2, 3])
    tf.debugging.assert_equal(tf.shape(y)[-1], NUM_CUERDAS)
    tf.debugging.assert_equal(tf.shape(temporal), tf.shape(y)[:-1])
    tf.debugging.assert_equal(tf.shape(por_cuerda), tf.shape(y))
    tf.debugging.assert_equal(tf.shape(pred), tf.concat([tf.shape(y), [NUM_CLASES_POR_CUERDA]], 0))
    tf.debugging.assert_greater_equal(y, tf.cast(-1, y.dtype))
    tf.debugging.assert_less(y, tf.cast(NUM_CLASES_POR_CUERDA, y.dtype))
    mask = temporal[..., None] & por_cuerda
    objetivos = tf.boolean_mask(y, mask)
    seleccion = tf.boolean_mask(pred, mask)
    tf.debugging.assert_greater_equal(objetivos, tf.cast(0, y.dtype),
                                      message="Etiqueta -1 sin máscara de exclusión")
    tf.debugging.assert_all_finite(seleccion, "Predicciones supervisadas no finitas")

    def calcular():
        if not from_logits:
            tf.debugging.assert_greater_equal(seleccion, tf.cast(0, pred.dtype))
            tf.debugging.assert_less_equal(seleccion, tf.cast(1, pred.dtype))
            tf.debugging.assert_near(tf.reduce_sum(seleccion, axis=-1),
                                     tf.ones(tf.shape(objetivos), dtype=pred.dtype), atol=1e-5)
        errores = tf.keras.losses.sparse_categorical_crossentropy(
            objetivos, seleccion, from_logits=from_logits)
        return tf.reduce_mean(errores)

    return tf.cond(tf.size(objetivos) > 0, calcular, lambda: tf.reduce_sum(pred[:0]))
