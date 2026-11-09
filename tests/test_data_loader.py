"""Pruebas sin descargar GuitarSet; se crean WAV/JAMS sintéticos."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcripcion_guitarra.data_loader import GuitarSetLoader, ErrorCargaGuitarSet  # noqa: E402


class TestGuitarSetLoader(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "guitarset"
        (self.root / "annotation").mkdir(parents=True)
        (self.root / "audio_mono-mic").mkdir(parents=True)

    def wav(self, track, folder=None, seconds=1):
        folder = folder or self.root / "audio_mono-mic"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{track}_mic.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\x00\x00" * int(seconds * 8000))
        return path

    def jams(self, track, duration=1.0, folder=None, valid=True):
        folder = folder or self.root / "annotation"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{track}.jams"
        path.write_text(json.dumps({"file_metadata": {"duration": duration}, "annotations": [{"namespace": "pitch_midi", "data": []}]}) if valid else "{", encoding="utf-8")
        return path

    def test_pareja_valida_y_exportacion(self):
        self.wav("00_ejemplo")
        self.jams("00_ejemplo")
        resultado = GuitarSetLoader(self.root).inventariar()
        self.assertEqual(resultado.resumen["parejas_incluidas"], 1)
        self.assertEqual(resultado.incidencias, [])
        registro = resultado.registros[0]
        self.assertEqual(registro.performer_group, "00")
        self.assertEqual(registro.wav_path, "audio_mono-mic/00_ejemplo_mic.wav")
        self.assertEqual(registro.duration_s, 1.0)
        m, i = resultado.guardar(Path(self.temp.name) / "reports")
        self.assertEqual(json.loads(m.read_text(encoding="utf-8"))["summary"]["parejas_incluidas"], 1)
        self.assertEqual(json.loads(i.read_text(encoding="utf-8"))["issues"], [])

    def test_parejas_incompletas(self):
        self.wav("00_sin_jams")
        self.jams("01_sin_wav")
        resultado = GuitarSetLoader(self.root).inventariar()
        self.assertEqual(resultado.resumen["parejas_incluidas"], 0)
        self.assertEqual({i.codigo for i in resultado.incidencias}, {"JAMS_FALTANTE", "WAV_FALTANTE"})

    def test_duplicados_ambiguos(self):
        self.wav("02_ejemplo")
        self.wav("02_ejemplo", self.root / "audio_mono-mic" / "copia")
        self.jams("02_ejemplo")
        resultado = GuitarSetLoader(self.root).inventariar()
        self.assertFalse(resultado.registros[0].included)
        self.assertIn("WAV_DUPLICADO", {i.codigo for i in resultado.incidencias})

    def test_jams_corrupto(self):
        self.wav("03_ejemplo")
        self.jams("03_ejemplo", valid=False)
        resultado = GuitarSetLoader(self.root).inventariar()
        self.assertIn("JAMS_INVALIDO", {i.codigo for i in resultado.incidencias})

    def test_duraciones_distintas(self):
        self.wav("04_ejemplo")
        self.jams("04_ejemplo", duration=2.0)
        resultado = GuitarSetLoader(self.root).inventariar()
        self.assertIn("DURACION_DIFERENTE", {i.codigo for i in resultado.incidencias})

    def test_sin_grupo(self):
        self.wav("ejemplo")
        self.jams("ejemplo")
        resultado = GuitarSetLoader(self.root).inventariar()
        self.assertIn("GRUPO_NO_IDENTIFICADO", {i.codigo for i in resultado.incidencias})

    def test_nombre_wav_invalido(self):
        pista = self.wav("05_ejemplo")
        pista.rename(pista.with_name("05_ejemplo.wav"))
        self.jams("05_ejemplo")
        r = GuitarSetLoader(self.root).inventariar()
        self.assertIn("NOMBRE_WAV_INVALIDO", {i.codigo for i in r.incidencias})
        self.assertEqual(r.resumen["parejas_incluidas"], 0)

    def test_jams_vacio(self):
        self.wav("05_ejemplo")
        p = self.jams("05_ejemplo")
        p.write_text(json.dumps({"file_metadata": {"duration": 1}, "annotations": []}), encoding="utf-8")
        r = GuitarSetLoader(self.root).inventariar()
        self.assertIn("JAMS_INVALIDO", {i.codigo for i in r.incidencias})

    def test_directorio_inexistente(self):
        with self.assertRaises(ErrorCargaGuitarSet):
            GuitarSetLoader(self.root / "no_existe").inventariar()


if __name__ == "__main__":
    unittest.main()
