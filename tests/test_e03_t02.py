"""E03-T02: pruebas de bordes y trazabilidad del preprocesamiento existente."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
# Si se ejecuta desde la carpeta del proyecto (tests/), usar src/ del proyecto.
PROJECT_ROOT = ROOT if (ROOT / 'src').exists() else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from transcripcion_guitarra.preprocessing import (  # noqa: E402
    AudioPreprocessor, SegmentadorAudio, ConfiguracionPreprocesamiento
)

SR = 8000
CFG = ConfiguracionPreprocesamiento(window_s=2.0, hop_s=1.0,
                                    peak_normalize=True, peak_target=0.95,
                                    pad_last=True)

class PruebasE03T02(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'ejemplo.wav'

    def guardar(self, datos):
        sf.write(self.path, np.asarray(datos, dtype=np.float32), SR, subtype='FLOAT')
        return self.path

    def procesar(self, datos, config=CFG):
        p = self.guardar(datos)
        a = AudioPreprocessor(config).procesar(p)
        return a, SegmentadorAudio(config).segmentar(a)

    def test_01_silencio_sin_nan_infinito(self):
        audio, segmentos = self.procesar(np.zeros(3 * SR))
        self.assertTrue(np.isfinite(audio.samples).all())
        self.assertTrue(np.all(audio.samples == 0))
        self.assertTrue(all(np.isfinite(s.samples).all() for s in segmentos))
        self.assertEqual(len(segmentos), 2)

    def test_02_silencio_no_division_por_cero(self):
        audio, _ = self.procesar(np.zeros(SR))
        self.assertIn('silencio_sin_division', audio.transformaciones)

    def test_03_audio_corto_relleno_y_tiempo(self):
        datos = np.sin(2*np.pi*220*np.arange(SR//2)/SR)
        audio, s = self.procesar(datos)
        self.assertEqual(len(s), 1)
        self.assertEqual(len(s[0].samples), 2*SR)
        self.assertEqual(s[0].valid_samples, SR//2)
        self.assertEqual((s[0].start_s,s[0].end_s), (0.0,0.5))
        self.assertTrue(s[0].padded)
        np.testing.assert_array_equal(s[0].samples[:SR//2], audio.samples)
        self.assertTrue(np.all(s[0].samples[SR//2:] == 0))

    def test_04_ultima_ventana_incompleta(self):
        datos = np.linspace(-.5,.5, int(2.3*SR),dtype=np.float32)
        audio, segmentos = self.procesar(datos)
        self.assertEqual(len(segmentos), 2)
        fin = segmentos[-1]
        self.assertEqual((fin.start_s, fin.end_s), (1.0,2.3))
        self.assertTrue(fin.padded)
        self.assertEqual(fin.valid_samples, int(1.3*SR))
        np.testing.assert_array_equal(fin.samples[:fin.valid_samples], audio.samples[SR:])
        self.assertTrue(np.all(fin.samples[fin.valid_samples:] == 0))

    def test_05_duracion_exacta_sin_relleno(self):
        _, segmentos = self.procesar(np.ones(2*SR)*.2)
        self.assertEqual(len(segmentos),1)
        self.assertFalse(segmentos[0].padded)
        self.assertEqual(segmentos[0].valid_samples,2*SR)

    def test_06_sin_relleno_descarta_incompleta(self):
        cfg = ConfiguracionPreprocesamiento(window_s=2,hop_s=1,pad_last=False)
        _, segmentos = self.procesar(np.ones(int(2.3*SR))*.2,cfg)
        self.assertEqual(len(segmentos),1)
        self.assertFalse(segmentos[0].padded)

    def test_07_normalizacion_pico_095(self):
        a,_ = self.procesar(np.sin(2*np.pi*220*np.arange(SR)/SR)*.25)
        self.assertAlmostEqual(float(np.max(np.abs(a.samples))),.95,places=4)

    def test_08_audio_estereo_a_mono(self):
        n=SR
        a,_ = self.procesar(np.stack([np.ones(n)*.2,np.ones(n)*.4],axis=1))
        self.assertEqual(a.samples.ndim,1)
        self.assertEqual(len(a.samples),n)
        self.assertIn('mono_promedio_2_canales',a.transformaciones)

    def test_09_ventanas_solapamiento_preserva_muestras(self):
        a,s = self.procesar(np.linspace(-.3,.4,3*SR,dtype=np.float32))
        self.assertEqual(len(s),2)
        self.assertEqual((s[0].start_s,s[1].start_s),(0.0,1.0))
        np.testing.assert_array_equal(s[0].samples[SR:],s[1].samples[:SR])

    def test_10_archivo_vacio(self):
        a,s = self.procesar(np.array([],dtype=np.float32))
        self.assertEqual(len(a.samples),0)
        self.assertEqual(len(s),1)
        self.assertEqual(s[0].valid_samples,0)
        self.assertTrue(s[0].padded)
        self.assertTrue(np.all(s[0].samples == 0))

if __name__ == '__main__':
    unittest.main()
