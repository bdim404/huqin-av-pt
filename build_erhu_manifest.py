import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


def audio_metadata(audio_path: Path) -> dict[str, float | int]:
    audio_info = sf.info(audio_path)
    return {
        "sample_rate": audio_info.samplerate,
        "frames": audio_info.frames,
        "duration_seconds": round(audio_info.duration, 6),
    }


def write_csv(output_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def singlept_records(dataset_root: Path) -> tuple[list[dict], list[dict]]:
    records = []
    counts = Counter()
    durations = defaultdict(float)
    audio_paths = sorted((dataset_root / "SinglePT").glob("Erhu-*/*/*.wav"))

    for audio_path in audio_paths:
        performer = audio_path.parent.parent.name
        technique = audio_path.parent.name
        metadata = audio_metadata(audio_path)
        records.append(
            {
                "audio_path": audio_path.relative_to(dataset_root).as_posix(),
                "performer": performer,
                "technique": technique,
                "clip_id": audio_path.stem,
                **metadata,
            }
        )
        counts[(performer, technique)] += 1
        durations[(performer, technique)] += metadata["duration_seconds"]

    distribution = []
    total_counts = Counter()
    total_durations = defaultdict(float)
    for (performer, technique), clip_count in sorted(counts.items()):
        duration_seconds = round(durations[(performer, technique)], 6)
        distribution.append(
            {
                "performer": performer,
                "technique": technique,
                "clip_count": clip_count,
                "duration_seconds": duration_seconds,
            }
        )
        total_counts[technique] += clip_count
        total_durations[technique] += duration_seconds

    for technique, clip_count in sorted(total_counts.items()):
        distribution.append(
            {
                "performer": "all",
                "technique": technique,
                "clip_count": clip_count,
                "duration_seconds": round(total_durations[technique], 6),
            }
        )

    return records, distribution


def annotation_metadata(annotation_path: Path) -> tuple[int, int, Counter]:
    label_counts = Counter()
    note_count = 0
    annotated_note_count = 0

    with annotation_path.open(newline="", encoding="utf-8") as annotation_file:
        reader = csv.DictReader(annotation_file)
        label_columns = [
            column_name
            for column_name in reader.fieldnames or []
            if column_name == "PT" or column_name.startswith("PT")
        ]
        for row in reader:
            note_count += 1
            if row.get("PT", "").strip():
                annotated_note_count += 1
            for label_column in label_columns:
                technique = row.get(label_column, "").strip() or "normal"
                label_counts[(label_column, technique)] += 1

    return note_count, annotated_note_count, label_counts


def excerpt_records(dataset_root: Path) -> tuple[list[dict], list[dict]]:
    records = []
    counts = Counter()
    annotation_paths = sorted((dataset_root / "Excerpts").glob("Erhu-*/*/*-PT.csv"))

    for annotation_path in annotation_paths:
        performer = annotation_path.parents[1].name
        piece = annotation_path.parent.name
        recording_name = annotation_path.name.removesuffix("-PT.csv")
        audio_path = annotation_path.with_name(f"{recording_name}.wav")
        if not audio_path.exists():
            raise FileNotFoundError(f"missing audio for annotation: {annotation_path}")
        metadata = audio_metadata(audio_path)
        note_count, annotated_note_count, label_counts = annotation_metadata(annotation_path)
        records.append(
            {
                "audio_path": audio_path.relative_to(dataset_root).as_posix(),
                "annotation_path": annotation_path.relative_to(dataset_root).as_posix(),
                "onset_path": annotation_path.with_name(f"{recording_name}-onset.csv")
                .relative_to(dataset_root)
                .as_posix(),
                "pitch_path": annotation_path.with_name(f"{recording_name}-pitch.csv")
                .relative_to(dataset_root)
                .as_posix(),
                "musicxml_path": annotation_path.with_name(f"{recording_name}.musicxml")
                .relative_to(dataset_root)
                .as_posix(),
                "performer": performer,
                "piece": piece,
                "note_count": note_count,
                "annotated_note_count": annotated_note_count,
                "normal_note_count": note_count - annotated_note_count,
                **metadata,
            }
        )
        for (label_level, technique), count in label_counts.items():
            counts[(performer, label_level, technique)] += count

    distribution = []
    total_counts = Counter()
    for (performer, label_level, technique), note_count in sorted(counts.items()):
        distribution.append(
            {
                "performer": performer,
                "label_level": label_level,
                "technique": technique,
                "note_count": note_count,
            }
        )
        total_counts[(label_level, technique)] += note_count

    for (label_level, technique), note_count in sorted(total_counts.items()):
        distribution.append(
            {
                "performer": "all",
                "label_level": label_level,
                "technique": technique,
                "note_count": note_count,
            }
        )

    return records, distribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    arguments = parser.parse_args()

    dataset_root = arguments.dataset_root.resolve()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    singlept_manifest, singlept_distribution = singlept_records(dataset_root)
    excerpt_manifest, excerpt_distribution = excerpt_records(dataset_root)

    write_csv(
        output_dir / "erhu_singlept_manifest.csv",
        [
            "audio_path",
            "performer",
            "technique",
            "clip_id",
            "sample_rate",
            "frames",
            "duration_seconds",
        ],
        singlept_manifest,
    )
    write_csv(
        output_dir / "erhu_singlept_technique_counts.csv",
        ["performer", "technique", "clip_count", "duration_seconds"],
        singlept_distribution,
    )
    write_csv(
        output_dir / "erhu_excerpt_manifest.csv",
        [
            "audio_path",
            "annotation_path",
            "onset_path",
            "pitch_path",
            "musicxml_path",
            "performer",
            "piece",
            "note_count",
            "annotated_note_count",
            "normal_note_count",
            "sample_rate",
            "frames",
            "duration_seconds",
        ],
        excerpt_manifest,
    )
    write_csv(
        output_dir / "erhu_excerpt_technique_counts.csv",
        ["performer", "label_level", "technique", "note_count"],
        excerpt_distribution,
    )

    summary = {
        "singlept_clip_count": len(singlept_manifest),
        "singlept_technique_count": len(
            {record["technique"] for record in singlept_manifest}
        ),
        "excerpt_count": len(excerpt_manifest),
        "excerpt_note_count": sum(record["note_count"] for record in excerpt_manifest),
        "excerpt_annotated_note_count": sum(
            record["annotated_note_count"] for record in excerpt_manifest
        ),
    }
    with (output_dir / "erhu_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote manifests to {output_dir}")


if __name__ == "__main__":
    main()
