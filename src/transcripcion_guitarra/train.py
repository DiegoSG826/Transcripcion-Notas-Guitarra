"""E06: Pipeline de entrenamiento POO para CRNNTranscriptor con caché CQT y guardado por época."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Iterable, Iterator

import numpy as np
import tensorflow as tf

from .cqt_extraction import (
    AlineadorEtiquetas,
    ConfiguracionCQT,
    ConstructorLotes,
    CQTExtractor,
    EtiquetasAlineadas,
    ResultadoCQT,
)
from .domain import NUM_CUERDAS
from .model_crnn import CRNNTranscriptor
from .preprocessing import (
    AudioPreparado,
    AudioPreprocessor,
    ConfiguracionPreprocesamiento,
    SegmentadorAudio,
)
from .supervision import contar_aciertos_enmascarados, entropia_cruzada_enmascarada


@dataclass(frozen=True)
class ConfiguracionEntrenamiento:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    clip_norm: float = 1.0
    seed: int = 2026
    max_pistas_por_epoca: int | None = None
    checkpoint_path: str = "reports/e06_entrenamiento/mejor_crnn.keras"
    historial_path: str = "reports/e06_entrenamiento/historial_e06.json"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs y batch_size deben ser positivos.")
        if self.learning_rate <= 0 or self.clip_norm <= 0:
            raise ValueError("learning_rate y clip_norm deben ser positivos.")
        if self.max_pistas_por_epoca is not None and self.max_pistas_por_epoca <= 0:
            raise ValueError("max_pistas_por_epoca debe ser positivo.")


class ProveedorLotesGuitarSet:
    """Genera lotes [B, T, F, 1] cacheando CQT y etiquetas en disco tras la primera lectura."""

    def __init__(
        self,
        data_root: str | Path,
        config_json_path: str | Path = "configs/etiquetas.json",
        repo_root: str | Path = ".",
        prep_config: ConfiguracionPreprocesamiento | None = None,
        cqt_config: ConfiguracionCQT | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config_path = (self.repo_root / config_json_path).resolve()

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"No existe el directorio de audio GuitarSet: {self.data_root}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"No existe el archivo de configuración: {self.config_path}")

        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else self.data_root.parent / "cache_cqt_e06"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cfg_doc = json.loads(self.config_path.read_text(encoding="utf-8"))
        fuentes = self.cfg_doc["fuentes"]

        self.manifiesto = json.loads((self.repo_root / fuentes["manifiesto"]).read_text(encoding="utf-8"))
        self.particiones = json.loads((self.repo_root / fuentes["particiones"]).read_text(encoding="utf-8"))
        eventos_doc = json.loads((self.repo_root / fuentes["eventos"]).read_text(encoding="utf-8"))

        self._rutas_wav: dict[str, Path] = {
            t["track_id"]: self.data_root / t["wav_path"]
            for t in self.manifiesto["tracks"]
            if t.get("included") and t.get("wav_path")
        }

        self._eventos_por_pista: dict[str, list[dict]] = defaultdict(list)
        for ev in eventos_doc["events"]:
            self._eventos_por_pista[ev["track_id"]].append(ev)

        self.preprocesador = AudioPreprocessor(prep_config)
        self.segmentador = SegmentadorAudio(prep_config)
        self.extractor_cqt = CQTExtractor(cqt_config)
        self.alineador = AlineadorEtiquetas()
        self.constructor = ConstructorLotes()

    def pistas_de(self, particion: str) -> list[str]:
        if particion not in self.particiones["splits"]:
            raise ValueError(f"Partición desconocida: {particion}")
        return [tid for tid in self.particiones["splits"][particion] if tid in self._rutas_wav]

    def _obtener_segmentos_pista(self, track_id: str) -> list[tuple[ResultadoCQT, EtiquetasAlineadas]]:
        archivo_cache = self.cache_dir / f"{track_id}.npz"
        if archivo_cache.is_file():
            with np.load(archivo_cache, allow_pickle=False) as z:
                cqts = z["cqts"]
                centers = z["centers"]
                labels = z["labels"]
                valids = z["valids"]
                lvalids = z["lvalids"]
            return [
                (
                    ResultadoCQT(cqts[i], centers[i], {"track_id": track_id, "cached": True}),
                    EtiquetasAlineadas(labels[i], valids[i], lvalids[i]),
                )
                for i in range(len(cqts))
            ]

        audio = self.preprocesador.procesar(self._rutas_wav[track_id])
        segmentos = self.segmentador.segmentar(audio)
        eventos_pista = self._eventos_por_pista.get(track_id, [])

        resultados: list[tuple[ResultadoCQT, EtiquetasAlineadas]] = []
        for seg in segmentos:
            audio_seg = AudioPreparado(
                samples=seg.samples,
                sample_rate=seg.sample_rate,
                duration_original_s=seg.end_s - seg.start_s,
                transformaciones=audio.transformaciones + ("segmentacion",),
                source=f"{track_id}@{seg.start_s:.2f}-{seg.end_s:.2f}s",
            )
            cqt_res = self.extractor_cqt.extraer(audio_seg)
            etiquetas = self.alineador.alinear(
                cqt_res.frame_center_s,
                eventos_pista,
                inicio_s=seg.start_s,
                fin_valido_s=seg.end_s,
            )
            resultados.append((cqt_res, etiquetas))

        if resultados:
            np.savez(
                archivo_cache,
                cqts=np.stack([r[0].cqt for r in resultados]),
                centers=np.stack([r[0].frame_center_s for r in resultados]),
                labels=np.stack([r[1].labels for r in resultados]),
                valids=np.stack([r[1].valid_mask for r in resultados]),
                lvalids=np.stack([r[1].label_valid_mask for r in resultados]),
            )
        return resultados

    def iterar_lotes(
        self,
        particion: str,
        batch_size: int,
        mezclar: bool = False,
        seed: int = 2026,
        max_pistas: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        pistas = list(self.pistas_de(particion))
        if mezclar:
            random.Random(seed).shuffle(pistas)
        if max_pistas is not None:
            pistas = pistas[:max_pistas]

        buffer_ejemplos = []
        for track_id in pistas:
            for ejemplo in self._obtener_segmentos_pista(track_id):
                buffer_ejemplos.append(ejemplo)
                if len(buffer_ejemplos) == batch_size:
                    if mezclar:
                        random.Random(seed).shuffle(buffer_ejemplos)
                    yield self.constructor.construir(buffer_ejemplos)
                    buffer_ejemplos.clear()

        if buffer_ejemplos:
            yield self.constructor.construir(buffer_ejemplos)


class EntrenadorCRNN:
    """Gestiona optimización enmascarada, evaluación por cuerda y guardado incremental."""

    def __init__(
        self,
        modelo: CRNNTranscriptor,
        config: ConfiguracionEntrenamiento | None = None,
    ) -> None:
        self.modelo = modelo
        self.config = config or ConfiguracionEntrenamiento()
        tf.keras.utils.set_random_seed(self.config.seed)
        self.optimizador = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

    def _paso_entrenamiento(self, lote: dict[str, np.ndarray]) -> tuple[float, int, bool]:
        mask = lote["supervision_mask"]
        evaluados = int(mask.sum())

        with tf.GradientTape() as tape:
            probs = self.modelo.predecir_probabilidades(lote["x"], training=True)
            loss = entropia_cruzada_enmascarada(
                lote["y"],
                probs,
                lote["valid_mask"],
                lote["label_valid_mask"],
                from_logits=False,
            )

        if evaluados == 0:
            return 0.0, 0, False

        grads = tape.gradient(loss, self.modelo.modelo.trainable_variables)
        grads_validos = [
            (tf.clip_by_norm(g, self.config.clip_norm), v)
            for g, v in zip(grads, self.modelo.modelo.trainable_variables)
            if g is not None
        ]
        if not grads_validos:
            return float(loss), evaluados, False

        self.optimizador.apply_gradients(grads_validos)
        return float(loss), evaluados, True

    def evaluar_lotes(self, lotes: Iterable[dict[str, np.ndarray]]) -> dict:
        perdida_ponderada = 0.0
        total_evaluados = 0
        total_aciertos = 0
        eval_cuerda = np.zeros(NUM_CUERDAS, dtype=np.int64)
        aciertos_cuerda = np.zeros(NUM_CUERDAS, dtype=np.int64)
        excluidos_cuerda = np.zeros(NUM_CUERDAS, dtype=np.int64)

        for lote in lotes:
            probs = self.modelo.predecir_probabilidades(lote["x"], training=False)
            loss = float(
                entropia_cruzada_enmascarada(
                    lote["y"],
                    probs,
                    lote["valid_mask"],
                    lote["label_valid_mask"],
                    from_logits=False,
                )
            )
            preds = tf.argmax(probs, axis=-1, output_type=tf.int32).numpy().astype(np.int16)
            stats = contar_aciertos_enmascarados(
                lote["y"], preds, lote["valid_mask"], lote["label_valid_mask"]
            )
            n = stats["evaluados"]
            perdida_ponderada += loss * n
            total_evaluados += n
            total_aciertos += stats["aciertos"]
            eval_cuerda += np.asarray(stats["evaluados_por_cuerda"], dtype=np.int64)
            aciertos_cuerda += np.asarray(stats["aciertos_por_cuerda"], dtype=np.int64)
            excluidos_cuerda += np.asarray(stats["excluidos_por_cuerda"], dtype=np.int64)

        loss_media = (perdida_ponderada / total_evaluados) if total_evaluados > 0 else None
        acc_global = (total_aciertos / total_evaluados) if total_evaluados > 0 else None
        acc_por_cuerda = [
            (int(a) / int(n)) if n > 0 else None
            for a, n in zip(aciertos_cuerda, eval_cuerda)
        ]

        return {
            "loss": loss_media,
            "accuracy_global": acc_global,
            "evaluados": total_evaluados,
            "aciertos": total_aciertos,
            "accuracy_por_cuerda": acc_por_cuerda,
            "evaluados_por_cuerda": eval_cuerda.tolist(),
            "excluidos_por_cuerda": excluidos_cuerda.tolist(),
        }

    def entrenar(
        self,
        proveedor_train: Iterable[dict[str, np.ndarray]] | ProveedorLotesGuitarSet,
        proveedor_val: Iterable[dict[str, np.ndarray]] | ProveedorLotesGuitarSet | None = None,
    ) -> dict:
        cfg = self.config
        mejor_val_loss = float("inf")
        historial: list[dict] = []
        ruta_hist = Path(cfg.historial_path)
        ruta_hist.parent.mkdir(parents=True, exist_ok=True)

        for epoca in range(1, cfg.epochs + 1):
            if isinstance(proveedor_train, ProveedorLotesGuitarSet):
                lotes_train = proveedor_train.iterar_lotes(
                    "train",
                    batch_size=cfg.batch_size,
                    mezclar=True,
                    seed=cfg.seed + epoca,
                    max_pistas=cfg.max_pistas_por_epoca,
                )
            else:
                lotes_train = proveedor_train

            perdida_acum = 0.0
            eval_acum = 0
            pasos_aplicados = 0
            pasos_omitidos = 0

            for lote in lotes_train:
                loss_val, n_eval, actualizado = self._paso_entrenamiento(lote)
                if actualizado:
                    perdida_acum += loss_val * n_eval
                    eval_acum += n_eval
                    pasos_aplicados += 1
                else:
                    pasos_omitidos += 1

            train_loss = (perdida_acum / eval_acum) if eval_acum > 0 else None

            val_metrics = None
            if proveedor_val is not None:
                if isinstance(proveedor_val, ProveedorLotesGuitarSet):
                    lotes_val = proveedor_val.iterar_lotes(
                        "validation",
                        batch_size=cfg.batch_size,
                        mezclar=False,
                        max_pistas=cfg.max_pistas_por_epoca,
                    )
                else:
                    lotes_val = proveedor_val
                val_metrics = self.evaluar_lotes(lotes_val)

            registro_epoca = {
                "epoca": epoca,
                "train_loss": train_loss,
                "train_evaluados": eval_acum,
                "pasos_aplicados": pasos_aplicados,
                "pasos_omitidos_sin_supervision": pasos_omitidos,
                "validation": val_metrics,
            }
            historial.append(registro_epoca)

            criterio = (
                val_metrics["loss"]
                if (val_metrics and val_metrics["loss"] is not None)
                else (train_loss if train_loss is not None else float("inf"))
            )
            if criterio < mejor_val_loss:
                mejor_val_loss = criterio
                self.modelo.guardar(cfg.checkpoint_path)

            reporte = {
                "schema_version": "1.0",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_entrenamiento": asdict(cfg),
                "config_modelo": asdict(self.modelo.config),
                "mejor_loss": mejor_val_loss if np.isfinite(mejor_val_loss) else None,
                "historial": historial,
            }
            ruta_hist.write_text(json.dumps(reporte, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            val_str = (
                f" | val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy_global']:.4f}"
                if (val_metrics and val_metrics["loss"] is not None)
                else ""
            )
            t_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
            print(
                f"[Época {epoca:02d}/{cfg.epochs:02d}] train_loss={t_str} "
                f"(pasos={pasos_aplicados}, omitidos={pasos_omitidos}){val_str}",
                flush=True,
            )

        return reporte