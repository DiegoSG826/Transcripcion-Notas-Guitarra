"""Verifica la política E02 en todas las pistas y exporta evidencia JSON/CSV.

La rejilla usa duración y sample rate del manifiesto; no recalcula audio/CQT
ni constituye una evaluación musical o de un modelo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcripcion_guitarra.cqt_extraction import AlineadorEtiquetas
from transcripcion_guitarra.label_policy import CatalogoEtiquetas
from transcripcion_guitarra.supervision import mascara_supervision


def verificar(catalogo, hop_length):
    if hop_length < 1:
        raise ValueError("hop_length debe ser positivo")
    detalles = []
    intervalos = [dict(r) for r in catalogo.intervalos]
    for tid, pista in sorted(catalogo.pistas.items()):
        sr = pista["sample_rate"]
        muestras = round(pista["duration_s"] * sr)
        tiempos = np.arange(muestras // hop_length + 1, dtype=np.float64) * hop_length / sr
        alineadas = catalogo.alinear_pista(tid, tiempos)
        # Referencia sin ambiguos solo para comprobar que el resto no cambia.
        referencia = AlineadorEtiquetas().alinear(
            tiempos, [e for e in catalogo.eventos_pista(tid) if e["label_class"] is not None],
            fin_valido_s=pista["duration_s"])
        ambiguos = np.zeros(alineadas.labels.shape, dtype=bool)
        for caso in (r for r in intervalos if r["track_id"] == tid):
            activo = ((tiempos >= caso["onset_s"]) & (tiempos < caso["offset_s"])
                      & alineadas.valid_mask)
            cuerda = caso["string_index"]
            ambiguos[activo, cuerda] = True
            caso["frames_en_rejilla"] = int(activo.sum())
            inicio, fin = caso["onset_s"], caso["offset_s"]
            sondas = np.array([inicio, (inicio + fin) / 2, np.nextafter(fin, inicio)])
            prueba = catalogo.alinear_pista(tid, sondas)
            if np.any(prueba.label_valid_mask[prueba.valid_mask, cuerda]):
                raise ValueError(f"Intervalo ambiguo sin máscara: {tid}")
        esperada = referencia.label_valid_mask & ~ambiguos
        if not np.array_equal(alineadas.label_valid_mask, esperada):
            raise ValueError(f"La máscara modifica posiciones ajenas a las exclusiones: {tid}")
        if (np.any(alineadas.labels[ambiguos] != -1)
                or not np.array_equal(alineadas.labels[~ambiguos], referencia.labels[~ambiguos])):
            raise ValueError(f"Se alteraron etiquetas fuera del intervalo ambiguo: {tid}")
        supervisadas = mascara_supervision(alineadas.labels, alineadas.valid_mask,
                                          alineadas.label_valid_mask)
        detalles.append({
            "track_id": tid, "particion": catalogo.particion_por_pista[tid],
            "sample_rate": sr, "frames": len(tiempos),
            "posiciones_cuerda_supervisadas": int(supervisadas.sum()),
            "posiciones_cuerda_ambiguas": int(ambiguos.sum()),
            "posiciones_cuerda_conflictos": int((~referencia.label_valid_mask
                                                   & referencia.valid_mask[:, None]).sum()),
            "frames_relleno": int((~alineadas.valid_mask).sum()),
        })
    return {
        "schema_version": "1.0", "procedencia": catalogo.procedencia,
        "resumen": catalogo.resumen(),
        "verificacion": {
            "estado": "correcto",
            "metodo": "centros_de_frame_sobre_metadatos_del_manifiesto_y_sondas_de_intervalos",
            "hop_length": hop_length,
            "pistas_verificadas": len(detalles),
            "intervalos_verificados": len(intervalos),
            "posiciones_cuerda_ambiguas": sum(r["posiciones_cuerda_ambiguas"] for r in detalles),
            "posiciones_cuerda_conflictos": sum(r["posiciones_cuerda_conflictos"] for r in detalles),
            "intervalos_sin_centro_en_rejilla": sum(r["frames_en_rejilla"] == 0 for r in intervalos),
            "alcance": "No recalcula CQT ni certifica corrección musical; no son métricas del modelo.",
        },
        "pistas": detalles,
        "intervalos": intervalos,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/etiquetas.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/e02_mascaras")
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--demo-npz", type=Path, help="Verificar también una CQT real creada por probar_cqt.py")
    args = parser.parse_args()
    catalogo = CatalogoEtiquetas.cargar(ROOT, args.config)
    resultado = verificar(catalogo, args.hop_length)
    if args.demo_npz:
        with np.load(args.demo_npz, allow_pickle=False) as demo:
            metadata = json.loads(str(demo["metadata"]))
            if metadata.get("etiquetas") != catalogo.procedencia:
                raise ValueError("La demo no corresponde a las fuentes actuales del catálogo")
            tid = metadata["track_id"]
            tiempos = demo["frame_center_s"]
            alineadas = catalogo.alinear_pista(tid, tiempos)
            mascara = mascara_supervision(alineadas.labels, alineadas.valid_mask, alineadas.label_valid_mask)
            for nombre, esperado in (("labels", alineadas.labels), ("valid_mask", alineadas.valid_mask),
                                     ("label_valid_mask", alineadas.label_valid_mask), ("supervision_mask", mascara)):
                if not np.array_equal(demo[nombre], esperado):
                    raise ValueError(f"Máscara o etiquetas incoherentes en demo: {nombre}")
            if demo["cqt"].shape != (len(tiempos), metadata["n_bins"]) or not np.isfinite(demo["cqt"]).all():
                raise ValueError("CQT de demo con dimensiones o valores inválidos")
            resultado["demo_cqt"] = {
                "estado": "correcto", "track_id": tid,
                "archivo": args.demo_npz.name,
                "sha256": hashlib.sha256(args.demo_npz.read_bytes()).hexdigest(),
                "forma_cqt": list(demo["cqt"].shape),
                "posiciones_cuerda_excluidas": int((~mascara & alineadas.valid_mask[:, None]).sum()),
                "procedencia": "CQT extraída de WAV local mediante scripts/probar_cqt.py",
            }
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "e02_mascaras_verificacion.json"
    json_path.write_text(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    csv_path = args.output / "e02_intervalos_enmascarados.csv"
    campos = ["track_id", "particion", "string_index", "string_conventional", "onset_s", "offset_s",
              "midi_value", "desviacion_semitonos", "motivo", "tratamiento",
              "indice_evento_exportado", "frames_en_rejilla"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultado["intervalos"])
    print(json.dumps(resultado["resumen"], ensure_ascii=False))
    print("Verificación:", json_path)
    print("Intervalos:", csv_path)


if __name__ == "__main__":
    main()
