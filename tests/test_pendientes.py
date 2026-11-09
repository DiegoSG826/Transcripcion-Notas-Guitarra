import sys,unittest,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import soundfile as sf
from transcripcion_guitarra.partitions import GeneradorParticiones
from transcripcion_guitarra.preprocessing import AudioPreprocessor, SegmentadorAudio, ConfiguracionPreprocesamiento
from transcripcion_guitarra.cqt_extraction import AlineadorEtiquetas, ConstructorLotes, ResultadoCQT

class TestParticiones(unittest.TestCase):
    def setUp(self):
        self.m=[{'track_id':f'{i:02d}_a','performer_group':f'{i:02d}','included':True} for i in range(6)]
    def test_disjuntas(self):
        r=GeneradorParticiones().generar(self.m)
        self.assertEqual(sum(r['counts'].values()),6)
        self.assertTrue(GeneradorParticiones.validar(r['splits']))
    def test_determinismo(self):
        g=GeneradorParticiones();self.assertEqual(g.generar(self.m),g.generar(self.m))
    def test_sin_grupo(self):
        self.m[0].pop('performer_group')
        with self.assertRaises(ValueError): GeneradorParticiones().generar(self.m)
    def test_duplicados(self):
        with self.assertRaises(ValueError): GeneradorParticiones().generar(self.m+[self.m[0]])
    def test_cruce(self):
        with self.assertRaises(ValueError): GeneradorParticiones.validar({'train':['a'],'validation':['a'],'test':[]})
    def test_exportar(self):
        r=GeneradorParticiones().generar(self.m)
        with tempfile.TemporaryDirectory() as d:
            p=GeneradorParticiones.guardar(r,d)
            self.assertEqual(json.loads(p.read_text())['counts'],r['counts'])

class TestAudio(unittest.TestCase):
    def _audio(self,x,sr=8000):
        d=tempfile.TemporaryDirectory();p=Path(d.name)/'a.wav';sf.write(p,x,sr);return d,p
    def test_silencio(self):
        d,p=self._audio(np.zeros(800));
        with d:
            a=AudioPreprocessor().procesar(p)
            self.assertTrue(np.isfinite(a.samples).all());self.assertEqual(np.max(abs(a.samples)),0)
    def test_mono(self):
        d,p=self._audio(np.ones((800,2))*0.1)
        with d:
            a=AudioPreprocessor().procesar(p)
            self.assertEqual(a.samples.ndim,1)
    def test_audio_corto_ventana(self):
        d,p=self._audio(np.ones(400)*0.2)
        with d:
            a=AudioPreprocessor().procesar(p)
            s=SegmentadorAudio().segmentar(a)
            self.assertEqual(len(s),1);self.assertTrue(s[0].padded);self.assertEqual(s[0].valid_samples,400)
    def test_ultima_ventana(self):
        cfg=ConfiguracionPreprocesamiento(window_s=.2,hop_s=.15)
        d,p=self._audio(np.ones(2400)*.2)
        with d:
            a=AudioPreprocessor(cfg).procesar(p)
            s=SegmentadorAudio(cfg).segmentar(a)
            self.assertTrue(s[-1].padded);self.assertAlmostEqual(s[-1].end_s,.3)
    def test_resample(self):
        d,p=self._audio(np.ones(800)*.2)
        with d:
            a=AudioPreprocessor(ConfiguracionPreprocesamiento(target_sample_rate=4000)).procesar(p)
            self.assertEqual(a.sample_rate,4000)

class TestCQT(unittest.TestCase):
    def test_semiabiertos_y_silencio(self):
        t=np.array([0.,.5,1.])
        r=AlineadorEtiquetas().alinear(t,[{'string_index':0,'onset_s':0.,'offset_s':1.,'label_class':1}])
        self.assertEqual(r.labels[:,0].tolist(),[1,1,0])
    def test_ambiguedad_no_es_silencio(self):
        r=AlineadorEtiquetas().alinear([0,.5,1],[{'string_index':1,'onset_s':0.,'offset_s':1.,'label_class':None}])
        self.assertEqual(r.labels[:,1].tolist(),[-1,-1,0]);self.assertFalse(r.label_valid_mask[0,1])
    def test_padding_lote(self):
        c1=ResultadoCQT(np.ones((2,3),dtype='float32'),np.array([0.,1.]),{})
        c2=ResultadoCQT(np.ones((3,3),dtype='float32'),np.array([0.,1.,2.]),{})
        a=AlineadorEtiquetas()
        r=ConstructorLotes().construir([(c1,a.alinear(c1.frame_center_s,[])),(c2,a.alinear(c2.frame_center_s,[]))])
        self.assertEqual(r['x'].shape,(2,3,3,1));self.assertFalse(r['valid_mask'][0,-1]);self.assertEqual(r['y'][0,-1,0],-1)

if __name__=='__main__':unittest.main()
