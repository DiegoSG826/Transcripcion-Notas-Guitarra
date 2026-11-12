"""Pruebas de E02-T02 con JAMS sintéticos reproducibles."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from transcripcion_guitarra.data_loader import ValidadorEtiquetas, ConfiguracionEtiquetas


class ValidacionEtiquetasTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'ejemplo.jams'
        self.anns = [dict(namespace='note_midi', annotation_metadata={'data_source': str(i)}, data=[])
                     for i in range(6)]
        self.val = ValidadorEtiquetas()

    def save(self):
        self.path.write_text(json.dumps({'annotations': self.anns}), encoding='utf-8')

    def test_seis_cuerdas_y_evento_valido(self):
        self.anns[0]['data'] = [{'time': 0, 'duration': 0.5, 'value': 40}]
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertTrue(r.apto_entrenamiento)
        self.assertEqual(len(r.eventos_validos), 1)
        self.assertEqual(r.eventos_validos[0].fret, 0)
        self.assertEqual(r.eventos[0].label_class, 1)
        self.assertEqual(r.eventos[0].string_conventional, 6)

    def test_traste_diecinueve(self):
        self.anns[5]['data'] = [{'time': .2, 'duration': .3, 'value': 83}]
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertEqual(r.eventos_validos[0].label_class, 20)
        self.assertEqual(r.eventos_validos[0].string_index, 5)

    def test_silencio_no_se_inventa_como_evento(self):
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertTrue(r.apto_entrenamiento)
        self.assertEqual(r.eventos, [])

    def test_falta_cuerda(self):
        self.anns.pop()
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba')
        self.assertIn('CUERDA_FALTANTE', [i.codigo for i in r.incidencias])

    def test_midi_desviacion_moderada_se_acepta_con_aviso(self):
        self.anns[0]['data'] = [{'time': 0, 'duration': 1, 'value': 40.31}]
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertEqual(len(r.eventos_validos), 1)
        self.assertTrue(r.apto_entrenamiento)
        self.assertEqual(r.eventos[0].estado_validacion, 'NORMALIZADO_ADVERTENCIA')
        self.assertEqual(r.eventos[0].midi_normalizado, 40)
        self.assertAlmostEqual(r.eventos[0].desviacion_semitonos, .31)
        self.assertTrue(any(i.severidad == 'advertencia' for i in r.incidencias))

    def test_midi_ambiguo_se_excluye_sin_convertir_en_silencio(self):
        self.anns[0]['data'] = [{'time': 0, 'duration': 1, 'value': 40.499}]
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertEqual(r.eventos_validos, [])
        self.assertFalse(r.apto_entrenamiento)
        self.assertIsNone(r.eventos[0].label_class)
        self.assertEqual(r.eventos[0].estado_validacion, 'INVALIDO')
        self.assertIn('MIDI_AMBIGUO', [i.codigo for i in r.incidencias])

    def test_borde_advertencia(self):
        self.anns[0]['data'] = [{'time': 0, 'duration': 1, 'value': 40.25}]
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertEqual(r.eventos[0].estado_validacion, 'NORMALIZADO')
        self.assertEqual(r.eventos[0].label_class, 1)

    def test_traste_fuera_rango(self):
        self.anns[0]['data'] = [{'time': 0, 'duration': 1, 'value': 65}]
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertIn('TRASTE_FUERA_RANGO', [i.codigo for i in r.incidencias])

    def test_tiempo_invalido(self):
        self.anns[0]['data'] = [{'time': 0, 'duration': 0, 'value': 40}]
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertIn('TIEMPO_INVALIDO', [i.codigo for i in r.incidencias])

    def test_tiempo_fuera_wav(self):
        self.anns[0]['data'] = [{'time': 1.9, 'duration': .5, 'value': 40}]
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertIn('FUERA_DURACION_WAV', [i.codigo for i in r.incidencias])

    def test_duplicado(self):
        self.anns[0]['data'] = [{'time': 0, 'duration': .5, 'value': 40}] * 2
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertIn('EVENTO_DUPLICADO', [i.codigo for i in r.incidencias])
        self.assertEqual(len(r.eventos_validos), 1)

    def test_fuente_duplicada(self):
        self.anns.append(self.anns[0].copy())
        self.save()
        r = self.val.validar_archivo(self.path, '00_prueba', 2)
        self.assertIn('CUERDA_DUPLICADA', [i.codigo for i in r.incidencias])

    def test_jams_corrupto(self):
        self.path.write_text('{', encoding='utf-8')
        r = self.val.validar_archivo(self.path, '00_prueba')
        self.assertIn('JAMS_NO_LEIBLE', [i.codigo for i in r.incidencias])

    def test_exportacion_y_manifiesto(self):
        self.save()
        manifest = Path(self.tmp.name) / 'manifest.json'
        manifest.write_text(json.dumps({'data_root': str(Path(self.tmp.name)), 'tracks': [
            {'included': True, 'track_id': '00_test', 'jams_path': 'ejemplo.jams', 'duration_s': 2}
        ]}), encoding='utf-8')
        r = self.val.validar_manifiesto(manifest)
        rutas = self.val.guardar(r, Path(self.tmp.name) / 'results')
        self.assertEqual(len(r), 1)
        self.assertTrue(all(path.exists() for path in rutas))
        self.assertEqual(json.loads(rutas[2].read_text())['summary']['pistas_procesadas'], 1)


if __name__ == '__main__':
    unittest.main()
