"""E02-T03: particiones reproducibles por intérprete antes de segmentar.
No modifica los datos originales ni los archivos de T01/T02.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import random
from collections import defaultdict


@dataclass(frozen=True)
class ConfiguracionParticiones:
    seed: int = 2026
    grupos_train: int = 4
    grupos_validation: int = 1
    grupos_test: int = 1


class GeneradorParticiones:
    def __init__(self, config: ConfiguracionParticiones | None = None):
        self.config = config or ConfiguracionParticiones()

    @staticmethod
    def _extraer_registros(manifiesto):
        if isinstance(manifiesto, (str, Path)):
            with open(manifiesto, encoding="utf-8") as f:
                manifiesto = json.load(f)
        if isinstance(manifiesto, list):
            return manifiesto
        for clave in ("tracks", "records", "registros", "manifest", "manifiesto"):
            if isinstance(manifiesto.get(clave), list):
                return manifiesto[clave]
        raise ValueError("No se encontró la lista de pistas del manifiesto")

    def generar(self, manifiesto):
        registros = self._extraer_registros(manifiesto)
        grupos = defaultdict(list)
        vistos = set()
        excluidos = []
        for r in registros:
            tid = r.get("track_id")
            if not isinstance(tid, str) or not tid:
                raise ValueError("Registro sin track_id")
            if tid in vistos:
                raise ValueError(f"track_id duplicado: {tid}")
            vistos.add(tid)
            if not r.get("included", True):
                excluidos.append(tid)
                continue
            grupo = r.get("performer_group") or r.get("performer") or r.get("player_id")
            if grupo is None or str(grupo).strip() == "":
                raise ValueError(f"Grupo de intérprete ausente en {tid}; detener para evitar fuga")
            grupos[str(grupo)].append(tid)
        n = len(grupos)
        cfg = self.config
        if cfg.grupos_train < 1 or cfg.grupos_validation < 1 or cfg.grupos_test < 1:
            raise ValueError("Cada partición requiere al menos un grupo")
        if n != cfg.grupos_train + cfg.grupos_validation + cfg.grupos_test:
            raise ValueError(f"Se esperaban {cfg.grupos_train+cfg.grupos_validation+cfg.grupos_test} grupos; hay {n}. Revisar metadatos o configuración")
        nombres = sorted(grupos)
        random.Random(cfg.seed).shuffle(nombres)
        asignacion = {
            "train": nombres[:cfg.grupos_train],
            "validation": nombres[cfg.grupos_train:cfg.grupos_train+cfg.grupos_validation],
            "test": nombres[cfg.grupos_train+cfg.grupos_validation:],
        }
        salida = {part: sorted(tid for g in gs for tid in grupos[g]) for part, gs in asignacion.items()}
        self.validar(salida, grupos, asignacion)
        return {
            "schema_version": "1.0", "seed": cfg.seed, "strategy": "group_by_performer_before_windowing",
            "groups": asignacion, "splits": salida, "excluded": sorted(excluidos),
            "counts": {k: len(v) for k,v in salida.items()},
        }

    @staticmethod
    def validar(splits, grupos=None, asignacion=None):
        conjuntos = {k:set(splits[k]) for k in ("train", "validation", "test")}
        for a,b in (("train","validation"),("train","test"),("validation","test")):
            cruces = conjuntos[a] & conjuntos[b]
            if cruces:
                raise ValueError(f"Fuga de pistas entre {a} y {b}: {sorted(cruces)[:5]}")
        if grupos is not None and asignacion is not None:
            for a,b in (("train","validation"),("train","test"),("validation","test")):
                if set(asignacion[a]) & set(asignacion[b]):
                    raise ValueError(f"Fuga de intérpretes entre {a} y {b}")
        return True

    @staticmethod
    def guardar(resultado, directorio):
        directorio = Path(directorio)
        directorio.mkdir(parents=True, exist_ok=True)
        for parte, ids in resultado["splits"].items():
            (directorio/f"{parte}.txt").write_text("\n".join(ids)+"\n", encoding="utf-8")
        (directorio/"particiones.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
        return directorio/"particiones.json"
