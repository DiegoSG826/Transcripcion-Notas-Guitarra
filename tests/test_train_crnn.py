import numpy as np

from transcripcion_guitarra.model_crnn import ConfiguracionCRNN, CRNNTranscriptor
from transcripcion_guitarra.train import ConfiguracionEntrenamiento, EntrenadorCRNN


def test_entrenador_crnn_reduce_perdida_y_omite_lote_vacio(tmp_path):
    rng = np.random.default_rng(123)
    x = rng.normal(size=(3, 24, 84, 1)).astype(np.float32)
    y = rng.integers(0, 21, size=(3, 24, 6), dtype=np.int16)
    valid = np.ones((3, 24), dtype=bool)
    valid[0, 18:] = False
    y[0, 18:] = -1
    lmask = np.broadcast_to(valid[..., None], y.shape).copy()

    lote_valido = {
        "x": x,
        "y": y,
        "valid_mask": valid,
        "label_valid_mask": lmask,
        "supervision_mask": lmask,
    }
    lote_vacio = {
        "x": x[:1],
        "y": np.full((1, 24, 6), -1, dtype=np.int16),
        "valid_mask": np.zeros((1, 24), dtype=bool),
        "label_valid_mask": np.zeros((1, 24, 6), dtype=bool),
        "supervision_mask": np.zeros((1, 24, 6), dtype=bool),
    }

    modelo = CRNNTranscriptor(ConfiguracionCRNN(conv_filters=(8, 16), lstm_units=16))
    ckpt = tmp_path / "mejor.keras"
    hist = tmp_path / "hist.json"
    cfg = ConfiguracionEntrenamiento(
        epochs=2,
        learning_rate=5e-3,
        checkpoint_path=str(ckpt),
        historial_path=str(hist),
    )

    entrenador = EntrenadorCRNN(modelo, cfg)
    reporte = entrenador.entrenar([lote_valido, lote_vacio], [lote_valido])

    assert len(reporte["historial"]) == 2
    assert reporte["historial"][0]["pasos_omitidos_sin_supervision"] == 1
    assert reporte["historial"][1]["train_loss"] < reporte["historial"][0]["train_loss"]
    assert ckpt.is_file()
    assert hist.is_file()