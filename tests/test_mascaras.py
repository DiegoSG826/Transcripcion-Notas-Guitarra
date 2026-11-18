"""Regresiones de exclusión por cuerda, cobertura de incidencias y aprendizaje."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcripcion_guitarra.cqt_extraction import AlineadorEtiquetas, ConstructorLotes, ResultadoCQT
from transcripcion_guitarra.label_policy import CatalogoEtiquetas
from transcripcion_guitarra.supervision import (
    mascara_supervision, contar_aciertos_enmascarados, entropia_cruzada_enmascarada,
)


def evento(cuerda=2, inicio=.5, fin=1., clase=None):
    return dict(string_index=cuerda, onset_s=inicio, offset_s=fin, label_class=clase,
                motivo_incidencia="MIDI_AMBIGUO" if clase is None else None)


class TestAlineacionMascaras(unittest.TestCase):
    def test_limites_semiabiertos_y_otras_cuerdas(self):
        tiempos = [np.nextafter(.5, 0), .5, .75, np.nextafter(1., 0), 1.]
        r = AlineadorEtiquetas().alinear(tiempos, [evento(), evento(0, 0, 2, 1)])
        self.assertEqual(r.labels[:, 2].tolist(), [0, -1, -1, -1, 0])
        self.assertTrue(r.valid_mask.all())
        self.assertTrue(r.label_valid_mask[:, [0, 1, 3, 4, 5]].all())
        self.assertTrue((r.labels[:, 0] == 1).all())  # Cuerda al aire, no silencio.

    def test_ambiguedad_y_conflicto_no_se_sobrescriben_segun_orden(self):
        eventos = [evento(), evento(2, 0, 2, 4), evento(2, .75, 1.5, 5)]
        resultados = [AlineadorEtiquetas().alinear([.25, .5, .8, 1., 1.5], lista)
                      for lista in itertools.permutations(eventos)]
        for r in resultados:
            np.testing.assert_array_equal(r.labels, resultados[0].labels)
            np.testing.assert_array_equal(r.label_valid_mask, resultados[0].label_valid_mask)

    def test_segmento_usa_tiempo_original_y_excluye_relleno(self):
        r = AlineadorEtiquetas().alinear([0, .25, .5, .75], [evento(2, 5.1, 5.3)],
                                         inicio_s=5., fin_valido_s=5.5)
        self.assertEqual(r.labels[:, 2].tolist(), [0, -1, -1, -1])
        self.assertEqual(r.valid_mask.tolist(), [True, True, False, False])
        self.assertTrue(r.label_valid_mask[:2, 0].all())
        self.assertFalse(r.label_valid_mask[2:].any())

    def test_otro_error_no_se_enmascara_como_ambiguedad(self):
        ev = evento()
        ev["motivo_incidencia"] = "TIEMPO_INVALIDO"
        with self.assertRaises(ValueError):
            AlineadorEtiquetas().alinear([0, 1], [ev])
        ev["motivo_incidencia"] = None
        with self.assertRaises(ValueError):
            AlineadorEtiquetas().alinear([0, 1], [ev])

    def test_no_mezclar_pistas(self):
        with self.assertRaises(ValueError):
            AlineadorEtiquetas().alinear([0], [dict(evento(), track_id="a"), dict(evento(), track_id="b")])

    def test_un_resumen_no_es_un_catalogo_de_silencios(self):
        with self.assertRaisesRegex(ValueError, "catálogo"):
            AlineadorEtiquetas().alinear([0], {"summary": {"eventos_ambiguos": 74}})

    def test_lote_conserva_mascara_por_cuerda_y_padding(self):
        alinear = AlineadorEtiquetas().alinear
        c1 = ResultadoCQT(np.ones((2, 3), dtype="float32"), np.array([0., .5]), {})
        c2 = ResultadoCQT(np.ones((3, 3), dtype="float32"), np.array([0., .5, 1.]), {})
        lote = ConstructorLotes().construir([(c1, alinear(c1.frame_center_s, [evento()])),
                                              (c2, alinear(c2.frame_center_s, []))])
        self.assertFalse(lote["supervision_mask"][0, 1, 2])
        self.assertTrue(lote["supervision_mask"][0, 1, 0])
        self.assertFalse(lote["supervision_mask"][0, 2].any())
        self.assertEqual(lote["y"][0, 1, 2], -1)


class TestCatalogo(unittest.TestCase):
    def setUp(self):
        self.manifiesto = {"tracks": [dict(track_id="00_a", included=True, performer_group="00",
                                          duration_s=4., sample_rate=8000)]}
        self.splits = {"splits": {"train": ["00_a"], "validation": [], "test": []}}
        self.eventos = [dict(evento(0, 0, .5, 1), track_id="00_a", midi_value=40., estado_validacion="NORMALIZADO"),
                        dict(evento(0, 1, 2), track_id="00_a", midi_value=40.495, estado_validacion="INVALIDO"),
                        dict(evento(0, 3, 3.5, 1), track_id="00_a", midi_value=40.3,
                             estado_validacion="NORMALIZADO_ADVERTENCIA")]
        self.incidencias = [dict(track_id="00_a", string_index=0, codigo="MIDI_AMBIGUO", severidad="error"),
                            dict(track_id="00_a", string_index=0, codigo="MIDI_DESVIACION_MODERADA",
                                 severidad="advertencia")]

    def cargar(self):
        return CatalogoEtiquetas(self.manifiesto, self.eventos, self.incidencias, self.splits)

    def test_conserva_ambiguos_y_advertencias(self):
        c = self.cargar()
        self.assertEqual(len(c.eventos_pista("00_a")), 3)
        self.assertEqual(len(c.intervalos), 1)
        self.assertEqual(c.advertencias, 1)
        self.assertEqual(c.resumen()["pistas_utilizables_con_mascaras"], 1)

    def test_borrar_ambiguo_no_lo_convierte_en_silencio(self):
        self.eventos.pop(1)
        with self.assertRaisesRegex(ValueError, "cobertura"):
            self.cargar()

    def test_error_estructural_sin_evento_bloquea(self):
        self.incidencias.append(dict(track_id="00_a", string_index=1, codigo="CUERDA_FALTANTE", severidad="error"))
        with self.assertRaisesRegex(ValueError, "cobertura"):
            self.cargar()

    def test_no_acepta_redondear_el_ambiguo(self):
        self.eventos[1]["label_class"] = 1
        with self.assertRaises(ValueError):
            self.cargar()

    def test_no_acepta_intervalo_no_localizable(self):
        self.eventos[1]["onset_s"] = None
        with self.assertRaises(ValueError):
            self.cargar()

    def test_no_acepta_pista_ausente_o_duplicada_en_splits(self):
        self.splits["splits"]["test"] = ["00_a"]
        with self.assertRaises(ValueError):
            self.cargar()

    def test_no_comparte_interpretes(self):
        self.manifiesto["tracks"].append(dict(self.manifiesto["tracks"][0], track_id="00_b"))
        self.splits["splits"]["test"] = ["00_b"]
        with self.assertRaises(ValueError):
            self.cargar()

    def test_fuentes_reales_conservan_las_360_pistas_y_74_intervalos(self):
        c = CatalogoEtiquetas.cargar(ROOT)
        self.assertEqual(len(c.pistas), 360)
        self.assertEqual(len(c.eventos), 62476)
        self.assertEqual(len(c.intervalos), 74)
        self.assertEqual(c.advertencias, 3311)
        for intervalo in c.intervalos:
            t = (intervalo["onset_s"] + intervalo["offset_s"]) / 2
            r = c.alinear_pista(intervalo["track_id"], [t])
            self.assertEqual(r.labels[0, intervalo["string_index"]], -1)
            self.assertFalse(r.label_valid_mask[0, intervalo["string_index"]])


class TestMetricas(unittest.TestCase):
    def test_no_cuenta_ambiguos_ni_padding_como_aciertos_o_errores(self):
        y = np.zeros((2, 6), dtype=np.int16)
        y[0, 2] = -1
        temporal = np.array([True, False])
        por_cuerda = np.ones((2, 6), dtype=bool)
        por_cuerda[0, 2] = False
        pred = y.copy()
        pred[0, 2] = 999
        pred[1] = 999
        r = contar_aciertos_enmascarados(y, pred, temporal, por_cuerda)
        self.assertEqual((r["evaluados"], r["aciertos"]), (5, 5))
        self.assertIsNone(r["exactitud_por_cuerda"][2])

    def test_rechaza_minus_uno_sin_mascara(self):
        with self.assertRaises(ValueError):
            mascara_supervision(np.full((1, 6), -1), np.ones(1, dtype=bool), np.ones((1, 6), dtype=bool))


class TestPerdidaTensorFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf

    def test_media_solo_sobre_objetivos_validos_y_gradientes_cero_en_excluidos(self):
        tf = self.tf
        y = np.full((1, 2, 6), -1, dtype=np.int16)
        y[0, 0, :2] = [0, 1]
        temporal = np.array([[True, False]])
        por_cuerda = y >= 0
        p = np.full((1, 2, 6, 21), np.nan, dtype=np.float32)
        p[0, 0, 0] = .2 / 20
        p[0, 0, 0, 0] = .8
        p[0, 0, 1] = .5 / 20
        p[0, 0, 1, 1] = .5
        pred = tf.Variable(p)
        with tf.GradientTape() as tape:
            perdida = entropia_cruzada_enmascarada(y, pred, temporal, por_cuerda)
        grad = tape.gradient(perdida, pred).numpy()
        self.assertAlmostEqual(float(perdida), float((-np.log(.8) - np.log(.5)) / 2), places=6)
        self.assertTrue((grad[~por_cuerda] == 0).all())
        self.assertTrue(np.isfinite(grad).all())

    def test_funciona_en_grafo_con_logits(self):
        tf = self.tf
        calcular = tf.function(lambda y, p, v, m: entropia_cruzada_enmascarada(y, p, v, m, from_logits=True))
        y = np.zeros((1, 2, 6), dtype=np.int16)
        v = np.array([[True, False]])
        m = np.ones(y.shape, dtype=bool)
        r = calcular(y, np.zeros((*y.shape, 21), dtype=np.float32), v, m)
        self.assertAlmostEqual(float(r), float(np.log(21)), places=6)

    def test_lote_sin_supervision_no_divide_por_cero(self):
        tf = self.tf
        y = np.full((1, 2, 6), -1, dtype=np.int16)
        p = tf.Variable(np.full((*y.shape, 21), np.nan, dtype=np.float32))
        with tf.GradientTape() as tape:
            perdida = entropia_cruzada_enmascarada(y, p, np.ones((1, 2), dtype=bool), np.zeros(y.shape, dtype=bool))
        self.assertEqual(float(perdida), 0.)
        self.assertTrue((tape.gradient(perdida, p).numpy() == 0).all())


if __name__ == "__main__":
    unittest.main()
