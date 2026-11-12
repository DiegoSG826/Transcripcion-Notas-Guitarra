from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from transcripcion_guitarra.preprocessing import AudioPreprocessor,SegmentadorAudio,ConfiguracionPreprocesamiento
p=argparse.ArgumentParser()
p.add_argument('wav',type=Path)
p.add_argument('--sample-rate',type=int,default=None)
p.add_argument('--window-s',type=float,default=2)
p.add_argument('--hop-s',type=float,default=1)
a=p.parse_args()
c=ConfiguracionPreprocesamiento(target_sample_rate=a.sample_rate,window_s=a.window_s,hop_s=a.hop_s)
x=AudioPreprocessor(c).procesar(a.wav);segs=SegmentadorAudio(c).segmentar(x)
print('SR:',x.sample_rate,'Duración original:',x.duration_original_s,'Muestras:',len(x.samples));print('Transformaciones:',x.transformaciones)
print('Ventanas:',len(segs));print('Primera:',(segs[0].start_s,segs[0].end_s,segs[0].padded) if segs else None)
print('Última:',(segs[-1].start_s,segs[-1].end_s,segs[-1].padded) if segs else None)
