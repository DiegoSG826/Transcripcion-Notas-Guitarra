#!/usr/bin/env python
"""
Auditoría complementaria de etiquetas GuitarSet para T02.
No modifica datos. Lee JAMS y genera un resumen JSON y un CSV de eventos.
Uso desde la raíz del repositorio:
  python scripts/auditar_etiquetas_t02.py ruta/a/annotation --salida reports/t02_etiquetas
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# GuitarSet/mirdata: 0 = cuerda 6 (E grave), 5 = cuerda 1 (E aguda).
# MIDI aproximado de las cuerdas al aire en afinación estándar.
SOURCE_INFO = {
    "0": {"string": 6, "open_midi": 40, "open_note": "E2"},
    "1": {"string": 5, "open_midi": 45, "open_note": "A2"},
    "2": {"string": 4, "open_midi": 50, "open_note": "D3"},
    "3": {"string": 3, "open_midi": 55, "open_note": "G3"},
    "4": {"string": 2, "open_midi": 59, "open_note": "B3"},
    "5": {"string": 1, "open_midi": 64, "open_note": "E4"},
}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_dir", help="Directorio con archivos .jams")
    parser.add_argument("--salida", default="reports/t02_etiquetas",
                        help="Prefijo/ruta base de los informes de salida")
    args = parser.parse_args()

    root = Path(args.annotation_dir)
    if not root.exists() or not root.is_dir():
        parser.error(f"No existe el directorio de anotaciones: {root}")

    jams_paths = sorted(root.rglob("*.jams"))
    if not jams_paths:
        parser.error(f"No se encontraron archivos .jams dentro de: {root}")

    counts = Counter()
    per_source = defaultdict(Counter)
    event_rows = []
    problems = []
    files_with_notes = set()
    all_values = []

    for path in jams_paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append({"file": str(path), "problem": f"json_error: {exc}"})
            continue

        track_id = str(doc.get("file_metadata", {}).get("title") or path.stem)
        annotations = [
            ann for ann in doc.get("annotations", [])
            if ann.get("namespace") == "note_midi"
        ]
        sources_seen = set()

        for ann in annotations:
            source = str(ann.get("annotation_metadata", {}).get("data_source", ""))
            if source not in SOURCE_INFO:
                problems.append({
                    "file": str(path), "track_id": track_id,
                    "problem": "unknown_data_source", "data_source": source
                })
                continue

            sources_seen.add(source)
            info = SOURCE_INFO[source]
            events = ann.get("data", [])
            counts["note_midi_annotations"] += 1
            per_source[source]["annotations"] += 1

            for idx, event in enumerate(events):
                try:
                    onset = float(event["time"])
                    duration = float(event["duration"])
                    midi = float(event["value"])
                except (KeyError, TypeError, ValueError):
                    problems.append({
                        "file": str(path), "track_id": track_id,
                        "problem": "malformed_event", "source": source,
                        "event_index": idx
                    })
                    counts["malformed_events"] += 1
                    continue

                if not all(math.isfinite(v) for v in (onset, duration, midi)):
                    counts["nonfinite_events"] += 1
                    problems.append({
                        "file": str(path), "track_id": track_id,
                        "problem": "nonfinite_event", "source": source,
                        "event_index": idx
                    })
                    continue

                raw_fret = midi - info["open_midi"]
                nearest_fret = int(round(raw_fret))
                deviation = abs(raw_fret - nearest_fret)
                in_range = 0 <= nearest_fret <= 19
                counts["events"] += 1
                per_source[source]["events"] += 1
                per_source[source]["fret_in_range_0_19"] += int(in_range)
                per_source[source]["fret_out_of_range"] += int(not in_range)
                per_source[source]["deviation_le_0_25"] += int(deviation <= 0.25)
                per_source[source]["deviation_gt_0_25"] += int(deviation > 0.25)
                per_source[source]["deviation_gt_0_5"] += int(deviation > 0.5)
                per_source[source][f"rounded_fret_{nearest_fret}"] += 1
                all_values.append(midi)
                files_with_notes.add(track_id)

                event_rows.append({
                    "track_id": track_id,
                    "file": str(path),
                    "data_source": source,
                    "string_conventional": info["string"],
                    "open_note_assumed": info["open_note"],
                    "event_index": idx,
                    "onset_s": onset,
                    "duration_s": duration,
                    "offset_s": onset + duration,
                    "midi_value": midi,
                    "raw_fret_estimate": round(raw_fret, 6),
                    "nearest_integer_fret": nearest_fret,
                    "distance_to_integer_semitone": round(deviation, 6),
                    "fret_in_0_19": in_range,
                    "duration_positive": duration > 0,
                    "onset_nonnegative": onset >= 0,
                })

        if len(sources_seen) != 6:
            problems.append({
                "file": str(path), "track_id": track_id,
                "problem": "note_midi_source_count_not_6",
                "sources_found": sorted(sources_seen)
            })

    output_base = Path(args.salida)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_base.with_suffix(".csv")
    json_path = output_base.with_suffix(".json")

    if event_rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()))
            writer.writeheader()
            writer.writerows(event_rows)

    summary = {
        "scope": "Preliminary structural and numeric audit; not proof of musical correctness.",
        "annotation_directory": str(root.resolve()),
        "jams_files_found": len(jams_paths),
        "tracks_with_note_events": len(files_with_notes),
        "source_mapping_reference": "mirdata GuitarSet loader: 0=low E/string 6, 5=high e/string 1",
        "open_string_midi_assumption": SOURCE_INFO,
        "counts": dict(counts),
        "per_data_source": {
            source: {
                "string_conventional": SOURCE_INFO[source]["string"],
                "open_note_assumed": SOURCE_INFO[source]["open_note"],
                "counts": dict(per_source[source]),
            }
            for source in sorted(SOURCE_INFO)
        },
        "midi_value_min": min(all_values) if all_values else None,
        "midi_value_max": max(all_values) if all_values else None,
        "problems_count": len(problems),
        "problems_sample": problems[:100],
        "interpretation_warning": (
            "raw_fret_estimate = MIDI value minus assumed open-string MIDI. "
            "Because note_midi values can be continuous, deviations can reflect pitch bends, "
            "intonation, annotation noise, or assumptions. Do not round these values blindly "
            "into final labels; inspect out-of-range and high-deviation events."
        ),
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"JAMS encontrados: {len(jams_paths)}")
    print(f"Eventos note_midi válidos: {counts['events']}")
    print(f"Problemas estructurales/numericos reportados: {len(problems)}")
    print(f"Resumen JSON: {json_path}")
    print(f"Detalle CSV: {csv_path}")
    print("NOTA: estimar traste con MIDI menos afinación al aire es una comprobación preliminar, no una etiqueta final.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
