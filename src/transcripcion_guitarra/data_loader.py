"""E02-T01: Inventario reproducible de pares WAV/JAMS de GuitarSet.

No valida eventos musicales ni crea particiones: E02-T02 y E02-T03
son responsables de esas operaciones.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import soundfile as sf


class ErrorCargaGuitarSet(Exception):
    """Error de configuración o de almacenamiento que impide inventariar."""


@dataclass(frozen=True)
class IncidenciaCarga:
    codigo: str
    track_id: str | None
    detalle: str
    ruta_relativa: str | None = None


@dataclass(frozen=True)
class RegistroGuitarSet:
    track_id: str
    wav_path: str | None
    jams_path: str | None
    duration_s: float | None
    performer_group: str | None
    included: bool
    exclusion_reason: str | None
    sample_rate: int | None = None
    channels: int | None = None
    jams_duration_s: float | None = None


@dataclass
class ResultadoCarga:
    root: str
    registros: list[RegistroGuitarSet] = field(default_factory=list)
    incidencias: list[IncidenciaCarga] = field(default_factory=list)

    @property
    def incluidos(self) -> list[RegistroGuitarSet]:
        return [r for r in self.registros if r.included]

    @property
    def resumen(self) -> dict[str, int]:
        return {
            "pistas": len(self.registros),
            "parejas_incluidas": len(self.incluidos),
            "pistas_excluidas": len(self.registros) - len(self.incluidos),
            "incidencias": len(self.incidencias),
        }

    def guardar(self, directorio: Path | str) -> tuple[Path, Path]:
        """Exporta manifiesto e incidencias en JSON (UTF-8)."""
        destino = Path(directorio)
        destino.mkdir(parents=True, exist_ok=True)
        manifiesto = destino / "e02_t01_manifiesto.json"
        incidencias = destino / "e02_t01_incidencias.json"
        fecha = datetime.now(timezone.utc).isoformat()
        cuerpo_manifiesto = {
            "schema_version": "1.0",
            "generated_at_utc": fecha,
            "data_root": self.root,
            "summary": self.resumen,
            "tracks": [asdict(r) for r in self.registros],
        }
        cuerpo_incidencias = {
            "schema_version": "1.0",
            "generated_at_utc": fecha,
            "summary": {"count": len(self.incidencias)},
            "issues": [asdict(i) for i in self.incidencias],
        }
        for path, content in ((manifiesto, cuerpo_manifiesto), (incidencias, cuerpo_incidencias)):
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return manifiesto, incidencias


class GuitarSetLoader:
    """Inventaría archivos y coordina lectores WAV/JAMS sin reglas musicales.

    Las rutas en el manifiesto son relativas a ``data_root``.
    No se infiere el intérprete de metadatos ambiguos: se utiliza el grupo
    de prefijo numérico ``NN_`` del nombre GuitarSet si está presente.
    """

    _PATRON_GRUPO = re.compile(r"^(\d{2})_")

    def __init__(self, data_root: Path | str, tolerance_s: float = 0.1) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        if not math.isfinite(tolerance_s) or tolerance_s < 0:
            raise ValueError("tolerance_s debe ser finita y no negativa.")
        self.tolerance_s = tolerance_s

    def _relativa(self, path: Path) -> str:
        return path.relative_to(self.data_root).as_posix()

    def _buscar(self, carpeta: Path, extension: str) -> dict[str, list[Path]]:
        encontrados: dict[str, list[Path]] = {}
        for ruta in sorted(carpeta.rglob("*")):
            if (not ruta.is_file() or ruta.suffix.lower() != extension
                    or "__MACOSX" in ruta.parts or ruta.name.startswith("._")):
                continue
            if extension == ".wav":
                track_id = ruta.stem[:-4] if ruta.stem.endswith("_mic") else ruta.stem
            else:
                track_id = ruta.stem
            encontrados.setdefault(track_id, []).append(ruta)
        return encontrados

    def _leer_wav(self, path: Path) -> tuple[float, int, int]:
        info = sf.info(path)
        if info.samplerate <= 0 or info.frames <= 0 or info.channels <= 0:
            raise ValueError("Cabecera WAV con sample rate, frames o canales inválidos")
        duration = info.frames / info.samplerate
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Duración WAV no finita o no positiva")
        return duration, int(info.samplerate), int(info.channels)

    def _leer_jams(self, path: Path) -> float | None:
        with path.open("r", encoding="utf-8") as archivo:
            doc = json.load(archivo)
        if not isinstance(doc, dict):
            raise ValueError("JAMS debe contener un objeto JSON")
        if not isinstance(doc.get("annotations"), list) or not doc["annotations"]:
            raise ValueError("JAMS no contiene lista annotations no vacía")
        for ann in doc["annotations"]:
            if not isinstance(ann, dict) or not isinstance(ann.get("namespace"), str) or not ann["namespace"]:
                raise ValueError("Una anotación JAMS no contiene namespace válido")
        metadata = doc.get("file_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("JAMS no contiene file_metadata válido")
        raw_duration: Any = metadata.get("duration")
        if raw_duration is None:
            return None
        if isinstance(raw_duration, bool) or not isinstance(raw_duration, (int, float)):
            raise ValueError("file_metadata.duration no es numérico")
        duration = float(raw_duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("file_metadata.duration no es finita y positiva")
        return duration

    def inventariar(self) -> ResultadoCarga:
        carpeta_wav = self.data_root / "audio_mono-mic"
        carpeta_jams = self.data_root / "annotation"
        for carpeta in (carpeta_wav, carpeta_jams):
            if not carpeta.is_dir():
                raise ErrorCargaGuitarSet(f"No existe el directorio requerido: {carpeta}")
        wavs = self._buscar(carpeta_wav, ".wav")
        jams = self._buscar(carpeta_jams, ".jams")
        resultado = ResultadoCarga(root=self.data_root.as_posix())

        for track_id in sorted(set(wavs) | set(jams)):
            wav_list = wavs.get(track_id, [])
            jams_list = jams.get(track_id, [])
            motivos: list[str] = []
            wav_path = self._relativa(wav_list[0]) if len(wav_list) == 1 else None
            jams_path = self._relativa(jams_list[0]) if len(jams_list) == 1 else None
            duration = None
            sample_rate = None
            channels = None
            jams_duration = None
            grupo = self._PATRON_GRUPO.match(track_id)
            performer_group = grupo.group(1) if grupo else None

            def anotar(codigo: str, detalle: str, ruta: str | None = None) -> None:
                motivos.append(codigo)
                resultado.incidencias.append(IncidenciaCarga(codigo, track_id, detalle, ruta))

            if len(wav_list) == 1 and not wav_list[0].stem.endswith("_mic"):
                anotar("NOMBRE_WAV_INVALIDO", "El WAV no cumple el patrón <track_id>_mic.wav", self._relativa(wav_list[0]))
            if not wav_list:
                anotar("WAV_FALTANTE", "No se encontró un <track_id>_mic.wav")
            elif len(wav_list) > 1:
                anotar("WAV_DUPLICADO", f"{len(wav_list)} candidatos: " + ", ".join(map(self._relativa, wav_list)))
            if not jams_list:
                anotar("JAMS_FALTANTE", "No se encontró un <track_id>.jams")
            elif len(jams_list) > 1:
                anotar("JAMS_DUPLICADO", f"{len(jams_list)} candidatos: " + ", ".join(map(self._relativa, jams_list)))
            if performer_group is None:
                anotar("GRUPO_NO_IDENTIFICADO", "El track_id no inicia con NN_; no se puede agrupar por intérprete")
            if wav_path is not None:
                try:
                    duration, sample_rate, channels = self._leer_wav(wav_list[0])
                    if channels != 1:
                        anotar("WAV_NO_MONO", f"Se esperaban 1 canal; se encontraron {channels}", wav_path)
                except (OSError, RuntimeError, ValueError) as exc:
                    anotar("WAV_INVALIDO", str(exc), wav_path)
            if jams_path is not None:
                try:
                    jams_duration = self._leer_jams(jams_list[0])
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                    anotar("JAMS_INVALIDO", str(exc), jams_path)
            if duration is not None and jams_duration is not None:
                if abs(duration - jams_duration) > self.tolerance_s:
                    anotar("DURACION_DIFERENTE", f"WAV={duration:.6f}s; JAMS={jams_duration:.6f}s", jams_path)
            resultado.registros.append(
                RegistroGuitarSet(
                    track_id=track_id,
                    wav_path=wav_path,
                    jams_path=jams_path,
                    duration_s=duration,
                    performer_group=performer_group,
                    included=not motivos,
                    exclusion_reason=";".join(motivos) or None,
                    sample_rate=sample_rate,
                    channels=channels,
                    jams_duration_s=jams_duration,
                )
            )
        return resultado

# ===============================================================
# E02-T02: validación de anotaciones de nota por cuerda (JAMS)
# ===============================================================
from .domain import (
    EventoNota, MIDI_CUERDAS_ABIERTAS, NUM_CUERDAS,
    TRASTE_MINIMO, TRASTE_MAXIMO,
)


@dataclass(frozen=True)
class ConfiguracionEtiquetas:
    """Umbrales en semitonos; las decisiones quedan registradas por evento.

    <= advertencia: normalizado, sin aviso.
    (advertencia, ambiguedad): normalizado con aviso.
    >= ambiguedad: ambiguo, SIN etiqueta para entrenamiento.
    """
    umbral_advertencia: float = 0.25
    umbral_ambiguedad: float = 0.49
    tolerancia_tiempo_s: float = 0.01

    def __post_init__(self) -> None:
        if not (math.isfinite(self.umbral_advertencia)
                and math.isfinite(self.umbral_ambiguedad)
                and 0 <= self.umbral_advertencia < self.umbral_ambiguedad < 0.5):
            raise ValueError('Se requiere 0 <= advertencia < ambiguedad < 0.5.')
        if not math.isfinite(self.tolerancia_tiempo_s) or self.tolerancia_tiempo_s < 0:
            raise ValueError('tolerancia_tiempo_s debe ser finita y no negativa.')


@dataclass(frozen=True)
class IncidenciaEtiqueta:
    track_id: str
    codigo: str
    detalle: str
    string_index: int | None = None
    event_index: int | None = None
    severidad: str = 'error'  # 'advertencia' no bloquea el entrenamiento


@dataclass(frozen=True)
class RegistroEvento:
    """Proveniencia y observaciones originales de un EventoNota.

    `evento` solo se construye cuando el traste está validado y dentro de [0,19].
    Los datos inválidos también se conservan en la exportación como registro
    con `evento=None` y `motivo_incidencia`.
    """
    track_id: str
    string_index: int
    string_conventional: int
    onset_s: float | None
    duration_s: float | None
    offset_s: float | None
    midi_value: float | None
    raw_fret_estimate: float | None
    label_class: int | None
    motivo_incidencia: str | None
    evento: EventoNota | None = None
    midi_normalizado: int | None = None
    desviacion_semitonos: float | None = None
    estado_validacion: str = 'INVALIDO'

    def para_json(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop('evento')  # ya representado en campos normalizados
        return data


@dataclass
class ResultadoEtiquetas:
    track_id: str
    eventos: list[RegistroEvento] = field(default_factory=list)
    incidencias: list[IncidenciaEtiqueta] = field(default_factory=list)

    @property
    def eventos_validos(self) -> list[EventoNota]:
        return [r.evento for r in self.eventos if r.evento is not None]

    @property
    def apto_entrenamiento(self) -> bool:
        """Solo los errores bloqueantes exigen revisión; avisos no la bloquean."""
        return not any(i.severidad == 'error' for i in self.incidencias)


class ValidadorEtiquetas:
    """Lee `note_midi` por cuerda y genera `EventoNota` válidos.

    No infiere silencios por ausencia de eventos: E04 asignará clase 0 a los
    frames no cubiertos por eventos. Tampoco aplica división train/val/test.
    """

    def __init__(self, configuracion: ConfiguracionEtiquetas | None = None) -> None:
        self.config = configuracion or ConfiguracionEtiquetas()

    @staticmethod
    def _numero(v: Any) -> float | None:
        if isinstance(v, bool) or not isinstance(v, (float, int)):
            return None
        value = float(v)
        return value if math.isfinite(value) else None

    def validar_archivo(self, jams_path: Path | str, track_id: str,
                       duracion_wav_s: float | None = None) -> ResultadoEtiquetas:
        """No altera archivos ni corrige JAMS: produce eventos + incidencias."""
        salida = ResultadoEtiquetas(track_id=track_id)

        def issue(codigo: str, detalle: str, cuerda: int | None = None,
                  indice: int | None = None, severidad: str = 'error') -> None:
            salida.incidencias.append(IncidenciaEtiqueta(track_id, codigo, detalle, cuerda, indice, severidad))

        path = Path(jams_path)
        try:
            with path.open('r', encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            issue('JAMS_NO_LEIBLE', str(exc))
            return salida
        if not isinstance(doc, dict) or not isinstance(doc.get('annotations'), list):
            issue('JAMS_ESTRUCTURA_INVALIDA', 'No contiene annotations como lista.')
            return salida

        if duracion_wav_s is not None and (
            self._numero(duracion_wav_s) is None or duracion_wav_s <= 0
        ):
            raise ValueError('duracion_wav_s debe ser positiva y finita.')

        notas = [a for a in doc['annotations'] if isinstance(a, dict) and a.get('namespace') == 'note_midi']
        fuentes: dict[int, dict[str, Any]] = {}
        for ann in notas:
            meta = ann.get('annotation_metadata')
            raw = meta.get('data_source') if isinstance(meta, dict) else None
            # JAMS/GuitarSet usa '0'...'5' como fuente por cuerda.
            if isinstance(raw, bool) or str(raw) not in {str(i) for i in range(NUM_CUERDAS)}:
                issue('CUERDA_DESCONOCIDA', f'data_source inválido: {raw!r}')
                continue
            cuerda = int(raw)
            if cuerda in fuentes:
                issue('CUERDA_DUPLICADA', f'Se repite note_midi de la cuerda {cuerda}', cuerda)
                fuentes[cuerda] = None  # invalida todas las anotaciones de esta cuerda
            elif cuerda not in fuentes:
                fuentes[cuerda] = ann
        for cuerda in range(NUM_CUERDAS):
            if cuerda not in fuentes:
                issue('CUERDA_FALTANTE', f'No existe note_midi para cuerda {cuerda}', cuerda)
                continue
            ann = fuentes[cuerda]
            if ann is None:
                continue
            data = ann.get('data')
            if not isinstance(data, list):
                issue('DATOS_NOTA_INVALIDOS', f'data no es lista para cuerda {cuerda}', cuerda)
                continue
            vistos: set[tuple[float, float, float]] = set()
            for idx, nota in enumerate(data):
                if not isinstance(nota, dict):
                    issue('EVENTO_INVALIDO', 'El elemento no es un objeto JSON', cuerda, idx)
                    continue
                inicio = self._numero(nota.get('time'))
                duracion = self._numero(nota.get('duration'))
                midi = self._numero(nota.get('value'))
                final = inicio + duracion if inicio is not None and duracion is not None else None
                traste_estimado = midi - MIDI_CUERDAS_ABIERTAS[cuerda] if midi is not None else None
                motivos: list[str] = []
                if inicio is None or duracion is None or inicio < 0 or duracion <= 0:
                    motivos.append('TIEMPO_INVALIDO')
                elif duracion_wav_s is not None and final > duracion_wav_s + self.config.tolerancia_tiempo_s:
                    motivos.append('FUERA_DURACION_WAV')
                if midi is None:
                    motivos.append('MIDI_INVALIDO')
                traste: int | None = None
                midi_normalizado: int | None = None
                desviacion: float | None = None
                advertencia = False
                if traste_estimado is not None:
                    # La afinación nominal es entera; el MIDI JAMS puede ser continuo.
                    candidato = round(midi)
                    midi_normalizado = int(candidato)
                    desviacion = abs(midi - candidato)
                    candidato_traste = candidato - MIDI_CUERDAS_ABIERTAS[cuerda]
                    if desviacion >= self.config.umbral_ambiguedad:
                        motivos.append('MIDI_AMBIGUO')
                    elif not TRASTE_MINIMO <= candidato_traste <= TRASTE_MAXIMO:
                        motivos.append('TRASTE_FUERA_RANGO')
                    else:
                        traste = int(candidato_traste)
                        advertencia = desviacion > self.config.umbral_advertencia
                if inicio is not None and duracion is not None and midi is not None:
                    clave = (inicio, duracion, midi)
                    if clave in vistos:
                        motivos.append('EVENTO_DUPLICADO')
                    vistos.add(clave)
                evento = None
                if not motivos:
                    evento = EventoNota(track_id=track_id, string_index=cuerda,
                                       onset_s=inicio, offset_s=final, fret=traste)
                detalle = (f'Evento {idx}, cuerda {cuerda}: tiempo={inicio}, '
                           f'duración={duracion}, MIDI={midi}, desviación={desviacion}')
                for codigo in motivos:
                    issue(codigo, detalle, cuerda, idx)
                if advertencia and not motivos:
                    issue('MIDI_DESVIACION_MODERADA', detalle, cuerda, idx,
                          severidad='advertencia')
                estado = ('INVALIDO' if motivos else
                          'NORMALIZADO_ADVERTENCIA' if advertencia else 'NORMALIZADO')
                salida.eventos.append(RegistroEvento(
                    track_id=track_id, string_index=cuerda,
                    string_conventional=NUM_CUERDAS - cuerda,
                    onset_s=inicio, duration_s=duracion, offset_s=final,
                    midi_value=midi, raw_fret_estimate=traste_estimado,
                    label_class=evento.label_class if evento else None,
                    motivo_incidencia=';'.join(motivos) or None,
                    evento=evento, midi_normalizado=midi_normalizado,
                    desviacion_semitonos=desviacion,
                    estado_validacion=estado,
                ))
        salida.eventos.sort(key=lambda x: (x.onset_s if x.onset_s is not None else float('inf'), x.string_index))
        return salida

    def validar_manifiesto(self, manifiesto_path: Path | str) -> list[ResultadoEtiquetas]:
        """Procesa solo pistas incluidas en T01 y utiliza rutas de su manifiesto."""
        path = Path(manifiesto_path)
        with path.open('r', encoding='utf-8') as fh:
            manifest = json.load(fh)
        root = Path(manifest['data_root'])
        resultados = []
        for pista in manifest['tracks']:
            if not pista['included']:
                continue
            relativa = Path(pista['jams_path'])
            if relativa.is_absolute() or '..' in relativa.parts:
                raise ValueError(f'Ruta insegura en manifiesto: {relativa}')
            resultados.append(self.validar_archivo(
                root / relativa, pista['track_id'], pista.get('duration_s')
            ))
        return resultados

    @staticmethod
    def guardar(resultados: list[ResultadoEtiquetas], carpeta: Path | str) -> tuple[Path, Path, Path]:
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        rutas = (
            carpeta / 'e02_t02_eventos.json',
            carpeta / 'e02_t02_incidencias.json',
            carpeta / 'e02_t02_resumen.json',
        )
        eventos = [r.para_json() for resultado in resultados for r in resultado.eventos]
        issues = [asdict(i) for resultado in resultados for i in resultado.incidencias]
        resumen = {
            'pistas_procesadas': len(resultados),
            'pistas_sin_incidencias': sum(r.apto_entrenamiento for r in resultados),
            'pistas_para_revision': sum(not r.apto_entrenamiento for r in resultados),
            'eventos_analizados': len(eventos),
            'eventos_validos': sum(len(r.eventos_validos) for r in resultados),
            'eventos_con_incidencias': sum(r.motivo_incidencia is not None for r in (e for resultado in resultados for e in resultado.eventos)),
            'incidencias': len(issues),
            'errores_bloqueantes': sum(i['severidad'] == 'error' for i in issues),
            'advertencias': sum(i['severidad'] == 'advertencia' for i in issues),
            'eventos_normalizados_con_advertencia': sum(e['estado_validacion'] == 'NORMALIZADO_ADVERTENCIA' for e in eventos),
            'eventos_ambiguos': sum('MIDI_AMBIGUO' in (e['motivo_incidencia'] or '') for e in eventos),
            'politica': 'MIDI entero más cercano; desviación <= 0.25 automática; (0.25,0.49) aviso; >= 0.49 ambiguo y excluido; originales conservados',
        }
        for ruta, contenido in zip(rutas, (
            {'schema_version': '1.0', 'events': eventos},
            {'schema_version': '1.0', 'issues': issues},
            {'schema_version': '1.0', 'summary': resumen},
        )):
            ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return rutas
