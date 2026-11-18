"""Catálogo de E02: fuente única de eventos completos y política de exclusión.

Valida también las incidencias: un JAMS no legible o una cuerda faltante no
pueden desaparecer de la supervisión como si sus notas fueran silencio.
No modifica anotaciones, informes históricos ni particiones.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .data_loader import ConfiguracionEtiquetas
from .domain import MIDI_CUERDAS_ABIERTAS, NUM_CUERDAS, TRASTE_MAXIMO


POLITICA = "enmascarar_intervalos_ambiguos_por_cuerda_v1"


class CatalogoEtiquetas:
    """Carga, comprueba y selecciona etiquetas sin perder eventos ambiguos."""

    @classmethod
    def cargar(cls, raiz: str | Path, config: str | Path | None = None):
        raiz = Path(raiz).resolve()
        ruta_config = Path(config) if config else raiz / "configs/etiquetas.json"
        if not ruta_config.is_absolute():
            ruta_config = raiz / ruta_config
        contenido_config = ruta_config.read_bytes()
        configuracion = json.loads(contenido_config)
        if configuracion.get("schema_version") != "1.0" or configuracion.get("politica") != POLITICA:
            raise ValueError("Configuración de política de etiquetas desconocida")
        cfg = ConfiguracionEtiquetas(**configuracion["normalizacion"])
        datos, fuentes = {}, {}
        for nombre in ("manifiesto", "eventos", "incidencias", "resumen", "particiones"):
            relativa = Path(configuracion["fuentes"][nombre])
            if relativa.is_absolute() or ".." in relativa.parts:
                raise ValueError("Las fuentes deben ser relativas a la raíz del proyecto")
            contenido = (raiz / relativa).read_bytes()
            datos[nombre] = json.loads(contenido)
            fuentes[nombre] = {"ruta": relativa.as_posix(), "sha256": hashlib.sha256(contenido).hexdigest()}
        catalogo = cls(datos["manifiesto"], datos["eventos"]["events"],
                       datos["incidencias"]["issues"], datos["particiones"], cfg)
        resumen = datos["resumen"]["summary"]
        comprobaciones = {
            "pistas_procesadas": len(catalogo.pistas),
            "eventos_analizados": len(catalogo.eventos),
            "eventos_validos": len(catalogo.eventos) - len(catalogo.intervalos),
            "eventos_ambiguos": len(catalogo.intervalos),
            "errores_bloqueantes": len(catalogo.intervalos),
            "advertencias": catalogo.advertencias,
            "incidencias": len(catalogo.incidencias),
        }
        if any(resumen.get(k) != v for k, v in comprobaciones.items()):
            raise ValueError("El resumen de T02 no corresponde a los eventos e incidencias")
        catalogo.procedencia = {
            "politica": POLITICA,
            "normalizacion": configuracion["normalizacion"],
            "config_sha256": hashlib.sha256(contenido_config).hexdigest(),
            "fuentes": fuentes,
        }
        return catalogo

    def __init__(self, manifiesto, eventos, incidencias, particiones,
                 configuracion: ConfiguracionEtiquetas | None = None):
        self.config = configuracion or ConfiguracionEtiquetas()
        self.procedencia = {"politica": POLITICA}
        registros = [r for r in manifiesto["tracks"] if r["included"]]
        self.pistas = {r["track_id"]: r for r in registros}
        if not registros or len(self.pistas) != len(registros):
            raise ValueError("Manifiesto vacío o con pistas duplicadas")
        self.particion_por_pista = {}
        interpretes = {}
        for parte in ("train", "validation", "test"):
            for tid in particiones["splits"][parte]:
                if tid in self.particion_por_pista or tid not in self.pistas:
                    raise ValueError(f"Pista duplicada o desconocida en particiones: {tid}")
                grupo = self.pistas[tid]["performer_group"]
                if not grupo or (grupo in interpretes and interpretes[grupo] != parte):
                    raise ValueError("Intérprete ausente o compartido entre particiones")
                interpretes[grupo] = parte
                self.particion_por_pista[tid] = parte
        if self.pistas.keys() != self.particion_por_pista.keys():
            raise ValueError("Las particiones no cubren exactamente las pistas incluidas")

        self.eventos = tuple(eventos)
        self.incidencias = tuple(incidencias)
        self._por_pista = defaultdict(list)
        self.intervalos = []
        esperadas = Counter()
        vistos = set()
        for indice, ev in enumerate(self.eventos):
            tid = ev["track_id"]
            cuerda = ev["string_index"]
            if tid not in self.pistas or type(cuerda) is not int or not 0 <= cuerda < NUM_CUERDAS:
                raise ValueError(f"Pista o cuerda inválida en evento {indice}")
            inicio, final, midi = (ev[k] for k in ("onset_s", "offset_s", "midi_value"))
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (inicio, final, midi)):
                raise ValueError(f"Evento sin tiempos/altura válidos: {tid}, {indice}")
            if inicio < 0 or final <= inicio or final > self.pistas[tid]["duration_s"] + self.config.tolerancia_tiempo_s:
                raise ValueError(f"Evento fuera de la duración válida: {tid}, {indice}")
            clave = (tid, cuerda, inicio, final, midi)
            if clave in vistos:
                raise ValueError(f"Evento duplicado: {tid}, {indice}")
            vistos.add(clave)
            entero = round(midi)
            desviacion = abs(midi - entero)
            traste = entero - MIDI_CUERDAS_ABIERTAS[cuerda]
            ambiguo = desviacion >= self.config.umbral_ambiguedad
            aviso = not ambiguo and desviacion > self.config.umbral_advertencia
            motivo = ev.get("motivo_incidencia")
            if ambiguo:
                if motivo != "MIDI_AMBIGUO" or ev.get("label_class") is not None:
                    raise ValueError(f"Ambigüedad sin exclusión explícita: {tid}, {indice}")
                if ev.get("estado_validacion") != "INVALIDO":
                    raise ValueError("El evento ambiguo debe conservar su estado original INVALIDO")
                esperadas[(tid, cuerda, "MIDI_AMBIGUO", "error")] += 1
                self.intervalos.append({
                    "indice_evento_exportado": indice, "track_id": tid,
                    "particion": self.particion_por_pista[tid],
                    "string_index": cuerda, "string_conventional": NUM_CUERDAS - cuerda,
                    "onset_s": inicio, "offset_s": final,
                    "midi_value": midi, "desviacion_semitonos": desviacion,
                    "motivo": motivo, "tratamiento": "enmascarar_por_cuerda",
                })
            else:
                estado = "NORMALIZADO_ADVERTENCIA" if aviso else "NORMALIZADO"
                clase = ev.get("label_class")
                if (not 0 <= traste <= TRASTE_MAXIMO or type(clase) is not int
                        or clase != traste + 1 or motivo or ev.get("estado_validacion") != estado):
                    raise ValueError(f"Etiqueta incompatible con la política: {tid}, {indice}")
                if aviso:
                    esperadas[(tid, cuerda, "MIDI_DESVIACION_MODERADA", "advertencia")] += 1
            self._por_pista[tid].append(ev)

        registradas = Counter((i["track_id"], i.get("string_index"), i["codigo"], i["severidad"])
                              for i in self.incidencias)
        if esperadas != registradas:
            raise ValueError("Incidencias sin cobertura o errores no enmascarables; revisar T02")
        self.advertencias = sum(i["severidad"] == "advertencia" for i in self.incidencias)

    def eventos_pista(self, track_id: str):
        """Devuelve TODOS los eventos de la pista, incluidos los ambiguos."""
        if track_id not in self.pistas:
            raise ValueError(f"Pista ausente del catálogo: {track_id}")
        return tuple(self._por_pista[track_id])

    def alinear_pista(self, track_id, frame_center_s, *, inicio_s=0.0, fin_valido_s=None):
        """Entrada de E04: conserva exclusiones y limita el tiempo al WAV real."""
        from .cqt_extraction import AlineadorEtiquetas

        eventos = self.eventos_pista(track_id)
        duracion = self.pistas[track_id]["duration_s"]
        fin = duracion if fin_valido_s is None else min(fin_valido_s, duracion)
        return AlineadorEtiquetas().alinear(frame_center_s, eventos,
                                            inicio_s=inicio_s, fin_valido_s=fin)

    def resumen(self):
        afectadas = {r["track_id"] for r in self.intervalos}
        return {
            "pistas_utilizables_con_mascaras": len(self.pistas),
            "pistas_con_intervalos_enmascarados": len(afectadas),
            "eventos_totales_conservados": len(self.eventos),
            "eventos_con_etiqueta": len(self.eventos) - len(self.intervalos),
            "intervalos_ambiguos": len(self.intervalos),
            "advertencias_conservadas": self.advertencias,
            "particiones": {
                p: {"pistas": sum(v == p for v in self.particion_por_pista.values()),
                    "intervalos_ambiguos": sum(r["particion"] == p for r in self.intervalos)}
                for p in ("train", "validation", "test")
            },
        }
