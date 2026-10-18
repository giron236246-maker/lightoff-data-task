"""Inspect the supplied CSV without inventing business conclusions."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def inspect_csv(csv_path, encoding=None):
    encodings = [encoding] if encoding else ["utf-8-sig", "gb18030"]
    last_error = None
    for candidate in encodings:
        try:
            frame = pd.read_csv(
                csv_path, encoding=candidate, dtype="string", keep_default_na=False
            )
            break
        except UnicodeError as error:
            last_error = error
    else:
        raise ValueError("无法识别 CSV 编码，请用 --encoding 指定。") from last_error

    columns = []
    for name in frame.columns:
        values = frame[name]
        blank = values.str.strip().eq("")
        columns.append({
            "name": str(name),
            "blank_count": int(blank.sum()),
            "blank_percent": round(float(blank.mean()) * 100, 2) if len(frame) else 0,
            "nonblank_unique_count": int(values[~blank].nunique()),
        })

    with csv_path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()

    return {
        "source_file": csv_path.name,
        "source_sha256": checksum,
        "encoding": candidate,
        "rows": len(frame),
        "column_count": len(frame.columns),
        "duplicate_rows_beyond_first": int(frame.duplicated().sum()),
        "blank_definition": "Empty string or whitespace only; literal NA/null retained.",
        "columns": columns,
    }


def main():
    parser = argparse.ArgumentParser(description="检查任务 CSV 的结构与数据质量")
    parser.add_argument("csv", type=Path, help="发放的 CSV 路径")
    parser.add_argument("--encoding", help="手动指定文件编码")
    parser.add_argument("--output", type=Path, default=Path("outputs/data_profile.json"))
    args = parser.parse_args()
    try:
        profile = inspect_csv(args.csv, args.encoding)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        parser.exit(1, f"读取失败：{error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"行数：{profile['rows']}，列数：{profile['column_count']}")
    print(f"完全重复记录（不含首条）：{profile['duplicate_rows_beyond_first']}")
    print(f"检查结果：{args.output.resolve()}")


if __name__ == "__main__":
    main()
