"""Build the document corpus from MSMARCO-XI parquet files.

Two things about this dataset drive the implementation, both verified against
the real file rather than assumed:

1. `datasets.load_dataset` FAILS on this repo with ArrowNotImplementedError
   ("Nested data conversions not implemented for chunked array outputs") —
   HF's own dataset-viewer is broken for it for the same reason. We read
   parquet directly and call .combine_chunks() before .to_pylist().

2. Each file has exactly ONE row group (97,941 rows in hinval.parquet), so
   there is no granularity for HTTP range reads — reading 3 rows over
   HfFileSystem took 41.8 s. Files are therefore downloaded whole via
   hf_hub_download (parallel + resumable + cached) and read from local disk.

Schema, confirmed:
    source_lang str            target_lang str  (FLORES code, e.g. "hin_Deva")
    query str                  Eng_Query str
    Answer str                 Eng_Answer str
    query_id int64             query_type str   (uppercase, e.g. "DESCRIPTION")
    meta struct{...}           <- translation decoding params; NOT useful
    passages struct{
        English_passages    list[str]    len 10
        Translated_passages list[str]    len 10
        is_selected         list[int]    len 10   <- exactly one 1 = gold
    }

Usage:
    python scripts/build_corpus.py --langs hi,ta --rows-per-lang 6000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.text import normalize  # noqa: E402

REPO = "ai4bharat/MSMARCO-XI"

# ISO-639-1 -> the dataset's file stem
LANG_FILE = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}

# Columns we actually need. `meta` is deliberately excluded — it is the
# translation model's decoding params and reading it wastes I/O on every row.
COLUMNS = [
    "query", "Eng_Query", "Answer", "Eng_Answer",
    "query_id", "query_type", "passages", "target_lang",
]

HELDOUT_FRACTION = 0.15


def download(lang: str) -> Path:
    stem = LANG_FILE[lang]
    return Path(
        hf_hub_download(REPO, f"validation/{stem}val.parquet", repo_type="dataset")
    )


def read_rows(path: Path, limit: int, batch_size: int = 512) -> Iterator[dict]:
    """Yield row dicts, projecting only COLUMNS.

    `.combine_chunks()` per batch is what dodges the nested-chunked-array
    conversion bug that breaks `datasets` on this repo.
    """
    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
        table = pa.Table.from_batches([batch]).combine_chunks()
        for row in table.to_pylist():
            yield row
            n += 1
            if n >= limit:
                return


def expand(row: dict, lang: str) -> Iterator[dict]:
    """One dataset row -> up to 20 documents (10 passages x {en, native}).

    `passages` is a struct of three PARALLEL LISTS, so the three are zipped.
    Treating it as a list of structs is the classic way to lose an hour here.
    """
    p = row.get("passages") or {}
    eng = p.get("English_passages") or []
    trans = p.get("Translated_passages") or []
    sel = p.get("is_selected") or []
    if not eng:
        return

    qid = row.get("query_id")
    qtype = (row.get("query_type") or "").strip().lower()   # "DESCRIPTION" -> "description"
    eng_query = normalize(row.get("Eng_Query") or "")
    src_query = normalize(row.get("query") or "")

    for i in range(len(eng)):
        is_sel = int(sel[i]) if i < len(sel) else 0
        for variant, text, dlang in (
            ("en", eng[i] if i < len(eng) else "", "en"),
            ("native", trans[i] if i < len(trans) else "", lang),
        ):
            text = normalize(text or "")
            if len(text) < 40:          # drop fragments that cannot carry an answer
                continue
            yield {
                "doc_id": f"{qid}:{lang}:{i}:{variant}",
                "text": text,
                "lang": dlang,
                "variant": variant,
                "query_id": qid,
                "query_type": qtype,
                "source_query": src_query,
                "eng_query": eng_query,
                "is_selected": is_sel,
                "answer": normalize(row.get("Answer") or "") or None,
                "eng_answer": normalize(row.get("Eng_Answer") or "") or None,
                "n_chars": len(text),
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,ta")
    ap.add_argument("--rows-per-lang", type=int, default=6000)
    ap.add_argument("--out", default=str(settings.corpus_dir))
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []
    heldout: list[dict] = []
    seen: set[str] = set()
    dupes = 0
    manifest: dict = {"repo": REPO, "langs": {}, "heldout_fraction": HELDOUT_FRACTION}

    for lang in langs:
        print(f"\n[{lang}] downloading (cached after first run)...")
        path = download(lang)
        print(f"[{lang}] reading {path.name} ({path.stat().st_size/1e6:.0f} MB)")

        n_rows = args.rows_per_lang
        cut = int(n_rows * (1 - HELDOUT_FRACTION))
        kept_before = len(docs)
        held_qids: list[int] = []

        for idx, row in enumerate(
            tqdm(read_rows(path, n_rows), total=n_rows, desc=f"[{lang}] rows", unit="row")
        ):
            # The held-out split is decided HERE, before anything is indexed.
            # Benchmark and eval queries must never have been seen by the index,
            # or the quality numbers are dishonest.
            if idx >= cut:
                held_qids.append(row.get("query_id"))
                heldout.append({
                    "query_id": row.get("query_id"),
                    "query": normalize(row.get("query") or ""),
                    "eng_query": normalize(row.get("Eng_Query") or ""),
                    "answer": normalize(row.get("Answer") or ""),
                    "eng_answer": normalize(row.get("Eng_Answer") or ""),
                    "query_type": (row.get("query_type") or "").strip().lower(),
                    "lang": lang,
                    "gold_doc_ids": [
                        f'{row.get("query_id")}:{lang}:{i}:{v}'
                        for i, s in enumerate(row.get("passages", {}).get("is_selected") or [])
                        if int(s) == 1 for v in ("en", "native")
                    ],
                })
                continue

            for doc in expand(row, lang):
                h = hashlib.sha1(doc["text"].encode("utf-8")).hexdigest()
                if h in seen:           # MS MARCO repeats passages across queries
                    dupes += 1
                    continue
                seen.add(h)
                docs.append(doc)

        manifest["langs"][lang] = {
            "file": path.name,
            "rows_read": min(n_rows, idx + 1),
            "indexed_rows": cut,
            "heldout_rows": len(held_qids),
            "docs_added": len(docs) - kept_before,
        }
        print(f"[{lang}] +{len(docs)-kept_before} docs, {len(held_qids)} held-out queries")

    table = pa.Table.from_pylist(docs)
    doc_path = out_dir / "documents.parquet"
    pq.write_table(table, doc_path, compression="zstd")

    held_path = settings.raw_dir / "heldout.jsonl"
    with held_path.open("w", encoding="utf-8") as f:
        for h in heldout:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    manifest["total_docs"] = len(docs)
    manifest["duplicates_dropped"] = dupes
    manifest["total_heldout"] = len(heldout)
    (settings.raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    by_lang: dict[str, int] = {}
    for d in docs:
        by_lang[d["lang"]] = by_lang.get(d["lang"], 0) + 1

    print(f"\n{'='*60}")
    print(f"documents      : {len(docs):,}  -> {doc_path}")
    print(f"  by language  : {by_lang}")
    print(f"  gold (is_sel): {sum(1 for d in docs if d['is_selected']):,}")
    print(f"duplicates     : {dupes:,} dropped")
    print(f"held-out       : {len(heldout):,} queries -> {held_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
