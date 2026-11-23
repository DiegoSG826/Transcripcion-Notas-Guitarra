import numpy as np
import tensorflow as tf

from transcripcion_guitarra.model_crnn import ConfiguracionCRNN, CRNNTranscriptor
from transcripcion_guitarra.supervision import (
    contar_aciertos_enmascarados,
    entropia_cruzada_enmascarada,
)


def test_crnn_preserva_tiempo_y_calcula_gradiente():
    modelo = CRNNTranscriptor(ConfiguracionCRNN(n_bins=84, conv_filters=(16, 32), lstm_units=32))

    # Probar dos longitudes temporales distintas (T=30 y T=75)
    for t_steps in (30, 75):
        x = np.random.randn(2, t_steps, 84, 1).astype(np.float32)
        probs = modelo.predecir_probabilidades(x, training=False)
        assert probs.shape == (2, t_steps, 6, 21)
        np.testing.assert_allclose(tf.reduce_sum(probs, axis=-1).numpy(), 1.0, atol=1e-5)

    # Verificar paso de gradiente con enmascaramiento de relleno (-1)
    x = tf.constant(np.random.randn(2, 40, 84, 1).astype(np.float32))
    y = np.random.randint(0, 21, (2, 40, 6), dtype=np.int16)
    valid_mask = np.ones((2, 40), dtype=bool)
    valid_mask[1, 30:] = False
    y[1, 30:] = -1
    label_mask = np.broadcast_to(valid_mask[..., None], y.shape).copy()

    with tf.GradientTape() as tape:
        preds_prob = modelo.predecir_probabilidades(x, training=True)
        loss = entropia_cruzada_enmascarada(y, preds_prob, valid_mask, label_mask)

    grads = tape.gradient(loss, modelo.modelo.trainable_variables)
    assert np.isfinite(float(loss))
    assert any(g is not None for g in grads)

    clases = modelo.predecir_clases(x)
    resumen = contar_aciertos_enmascarados(y, clases, valid_mask, label_mask)
    assert resumen["evaluados"] == (40 + 30) * 6