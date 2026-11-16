"""Summarize a tiny_voca.train JSONL log: progress, speed and the losses that matter.

    python scripts/parse_train_log.py logs_train_full.txt
    python scripts/parse_train_log.py logs_train_full.txt --every 1000 --validation
"""
import argparse
import json

KEYS = ("mel", "duration", "voicing", "pitch", "pause", "energy", "stft")
VALIDATION = ("validation_predicted_mel", "validation_duration_length_ratio",
              "validation_pitch_error_semitones", "validation_voicing_accuracy",
              "validation_pause_accuracy")


def rows(path):
    # The driver writes its own markers as UTF-8 and the appended python output as
    # UTF-16LE, so decode as UTF-8 and drop the NUL bytes of the UTF-16 records.
    with open(path, "rb") as stream:
        text = stream.read().decode("utf-8", errors="ignore").replace("\x00", "")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"step"' in line:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--every", type=int, default=0, help="Print one row per N steps")
    parser.add_argument("--validation", action="store_true", help="Print validation metrics too")
    parser.add_argument("--tail", type=int, help="Only the last N records")
    arguments = parser.parse_args()

    records = list(rows(arguments.log))
    if not records:
        raise SystemExit("no training records in " + arguments.log)
    first, last = records[0], records[-1]
    print(f"run: start {first['step']} -> last {last['step']} ({last['elapsed_seconds']:.0f} s, "
          f"{last['steps_per_second']:.2f} steps/s, "
          f"{last['frames'] / 93.75 / max(1e-9, last['elapsed_seconds'] - (records[-2]['elapsed_seconds'] if len(records) > 1 else 0)):.0f} audio-s/s)")
    print("step     loss    mel  durat  voice  pitch  pause   stft   teacher  gan  inst-steps/s  audio-s/s  batch-s")
    window = 0
    previous = None
    for record in records:
        window += 1
        rate, audio, batch = "", "", ""
        if previous is not None and record["elapsed_seconds"] > previous["elapsed_seconds"]:
            # "frames"/"utterances" are summed over the log window, whose length is the
            # gap between two records, so the window rate is frames / gap.
            seconds = record["elapsed_seconds"] - previous["elapsed_seconds"]
            steps = record["step"] - previous["step"]
            rate = f"{steps / seconds:11.2f}"
            audio = f"{record['frames'] / 93.75 / seconds:9.0f}"
            if steps:
                batch = f"{record['frames'] / steps / 93.75:7.1f}"
        previous = record
        if arguments.tail:
            if record is not records[-1]:
                continue
        elif arguments.every and window % arguments.every:
            continue
        elif not arguments.every and record["step"] % 1000:
            continue
        cells = " ".join(f"{record.get(key, float('nan')):6.3f}" for key in KEYS)
        print(f"{record['step']:6d} {record['loss']:8.2f} {cells} "
              f"{record.get('teacher_fraction', 0):7.2f} {record.get('gan_scale', 0):6.1f} {rate:>13} {audio:>9} {batch:>8}")
        if arguments.validation and "validation_predicted_mel" in record:
            print("        validation " + "  ".join(
                f"{key.replace('validation_', '')}={record[key]:.4f}" for key in VALIDATION
                if key in record))
    if arguments.tail:
        record = records[-1]
        print(json.dumps({key: record[key] for key in record if key.startswith("validation")},
                         indent=2))


if __name__ == "__main__":
    main()
