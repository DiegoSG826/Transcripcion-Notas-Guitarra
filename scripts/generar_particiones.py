from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from transcripcion_guitarra.partitions import GeneradorParticiones,ConfiguracionParticiones
p=argparse.ArgumentParser()
p.add_argument('--manifest',default='reports/e02_t01/e02_t01_manifiesto.json')
p.add_argument('--output',default='splits')
p.add_argument('--seed',type=int,default=2026)
a=p.parse_args()
g=GeneradorParticiones(ConfiguracionParticiones(seed=a.seed))
r=g.generar(ROOT/a.manifest if not Path(a.manifest).is_absolute() else a.manifest)
out=g.guardar(r,ROOT/a.output)
print('Intérpretes:',r['groups']);print('Pistas:',r['counts']);print('Excluidas:',len(r['excluded']));print('Sin intersecciones: OK');print('Archivo:',out)
