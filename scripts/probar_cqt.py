from pathlib import Path
import argparse,sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from transcripcion_guitarra.preprocessing import AudioPreprocessor,ConfiguracionPreprocesamiento
from transcripcion_guitarra.cqt_extraction import CQTExtractor,ConfiguracionCQT,AlineadorEtiquetas
p=argparse.ArgumentParser();p.add_argument('wav',type=Path);p.add_argument('--events',type=Path,default=None);p.add_argument('--output',type=Path,default=Path('reports/e04_demo/cqt_demo.npz'));a=p.parse_args()
x=AudioPreprocessor(ConfiguracionPreprocesamiento(peak_normalize=False)).procesar(a.wav)
c=CQTExtractor(ConfiguracionCQT()).extraer(x)
print('CQT:',c.cqt.shape,'SR:',c.metadata['sample_rate'],'Tiempo primer/último frame:',c.frame_center_s[0],c.frame_center_s[-1])
if a.events:
    obj=json.loads(a.events.read_text(encoding='utf-8'))
    # No alinear todos los eventos de GuitarSet contra un solo WAV
    track=a.wav.stem.removesuffix('_mic')
    items=obj.get('events',obj) if isinstance(obj,dict) else obj
    events=[e for e in items if e.get('track_id')==track]
    e=AlineadorEtiquetas().alinear(c.frame_center_s,events)
    print('Etiquetas:',e.labels.shape,'Frames/cuerda ambiguos:',int((~e.label_valid_mask).sum()))
else:
    e=AlineadorEtiquetas().alinear(c.frame_center_s,[])
out=a.output if a.output.is_absolute() else ROOT/a.output
out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out,cqt=c.cqt,frame_center_s=c.frame_center_s,labels=e.labels,valid_mask=e.valid_mask,label_valid_mask=e.label_valid_mask,metadata=json.dumps(c.metadata))
print('Archivo:',out)
