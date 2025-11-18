#!/usr/bin/env python3
"""Precompute Qwen3 embeddings for MIMIC readmission notes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from text_encoder import (
    Qwen3Embedding_0_6B_Encoder,
    Qwen3Embedding_4B_Encoder,
    Qwen3Embedding_8B_Encoder,
    Qwen3LLMEncoder,
)

import csv
import sys

csv.field_size_limit(sys.maxsize)

ModelFactory = Type[Qwen3LLMEncoder]

MODEL_ALIASES: Dict[str, ModelFactory] = {
    "qwen3-0.6b": Qwen3Embedding_0_6B_Encoder,
    "0.6b": Qwen3Embedding_0_6B_Encoder,
    "qwen/qwen3-embedding-0.6b": Qwen3Embedding_0_6B_Encoder,
    "qwen3-4b": Qwen3Embedding_4B_Encoder,
    "4b": Qwen3Embedding_4B_Encoder,
    "qwen/qwen3-embedding-4b": Qwen3Embedding_4B_Encoder,
    "qwen3-8b": Qwen3Embedding_8B_Encoder,
    "8b": Qwen3Embedding_8B_Encoder,
    "qwen/qwen3-embedding-8b": Qwen3Embedding_8B_Encoder,
}

DEFAULT_INSTRUCTION = (
    "Summarize the clinical note with emphasis on details relevant to thirty-day readmission risk."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute embeddings for MIMIC readmission notes")
    parser.add_argument("--csv-path", type=Path, default=Path("mimic_readmission.csv"), help="Input CSV path")
    parser.add_argument("--text-column", type=str, default="text", help="Column containing note text")
    parser.add_argument(
        "--label-column",
        type=str,
        default="race",
        help="Column containing multi-class race labels",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-0.6b",
        choices=sorted(set(MODEL_ALIASES.keys())),
        help="Qwen3 embedding model alias",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=8192,
        help="Token limit passed to the encoder",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=2048,
        help="Number of rows to stream from CSV per chunk",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of rows to embed (useful for smoke tests)",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Override the detected number of rows (skips counting pass)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction prepended to each note before encoding (set to empty string to disable)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="CSV file encoding",
    )
    parser.add_argument(
        "--csv-engine",
        type=str,
        default="python",
        choices=["c", "python"],
        help="Engine passed to pandas.read_csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store embeddings and metadata",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="qwen3_embeddings",
        help="Filename prefix for produced artifacts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs",
    )
    return parser.parse_args()


def resolve_model(model_key: str) -> ModelFactory:
    normalized = model_key.lower()
    if normalized not in MODEL_ALIASES:
        known = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Model `{model_key}` not supported. Available aliases: {known}")
    return MODEL_ALIASES[normalized]


def iter_chunks(
    csv_path: Path,
    text_column: str,
    label_column: str,
    chunk_size: int,
    encoding: str,
    engine: str,
    limit: Optional[int],
) -> Iterator[pd.DataFrame]:
    rows_yielded = 0
    reader = pd.read_csv(
        csv_path,
        usecols=[text_column, label_column],
        chunksize=chunk_size,
        encoding=encoding,
        engine=engine,
        keep_default_na=False,
        dtype={text_column: str},
    )
    for chunk in reader:
        if limit is not None and rows_yielded >= limit:
            break
        if limit is not None:
            rows_remaining = limit - rows_yielded
            if rows_remaining <= 0:
                break
            chunk = chunk.iloc[:rows_remaining]
        rows_yielded += len(chunk)
        yield chunk


def count_rows(
    csv_path: Path,
    text_column: str,
    chunk_size: int,
    encoding: str,
    engine: str,
    limit: Optional[int],
) -> int:
    if limit is not None:
        return limit
    total = 0
    reader = pd.read_csv(
        csv_path,
        usecols=[text_column],
        chunksize=chunk_size,
        encoding=encoding,
        engine=engine,
        keep_default_na=False,
    )
    for chunk in reader:
        total += len(chunk)
    return total


def ensure_outputs(output_dir: Path, prefix: str, overwrite: bool) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / f"{prefix}.feather"
    labels_path = output_dir / f"{prefix}_labels.feather"
    metadata_path = output_dir / f"{prefix}_metadata.json"
    if not overwrite:
        for path in (embedding_path, labels_path, metadata_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    return embedding_path, labels_path, metadata_path


def embeddings_to_batch(
    embeddings: np.ndarray,
    embedding_dim: int,
) -> pa.RecordBatch:
    if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"Expected embeddings with shape (batch, {embedding_dim}), got {embeddings.shape}"
        )
    flattened = embeddings.reshape(-1)
    values = pa.array(flattened, type=pa.float32())
    list_array = pa.FixedSizeListArray.from_arrays(values, embedding_dim)
    return pa.RecordBatch.from_arrays([list_array], names=["embedding"])


def prepare_inputs(raw_texts: List[str], encoder: Qwen3LLMEncoder, instruction: Optional[str]) -> List[str]:
    if instruction is None or instruction.strip() == "":
        return raw_texts
    return [encoder.add_instruction(instruction, text) for text in raw_texts]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    encoder_cls = resolve_model(args.model)
    logging.info("Loading encoder %s", encoder_cls.__name__)
    encoder = encoder_cls(max_input_length=args.max_input_length)

    total_rows = args.num_rows or count_rows(
        csv_path=args.csv_path,
        text_column=args.text_column,
        chunk_size=args.read_batch_size,
        encoding=args.encoding,
        engine=args.csv_engine,
        limit=args.limit,
    )
    if total_rows <= 0:
        raise ValueError("No rows to encode")
    logging.info("Planning to encode %d rows", total_rows)

    embedding_path, labels_path, metadata_path = ensure_outputs(
        output_dir=args.output_dir,
        prefix=args.output_prefix,
        overwrite=args.overwrite,
    )

    chunk_iter = iter_chunks(
        csv_path=args.csv_path,
        text_column=args.text_column,
        label_column=args.label_column,
        chunk_size=args.read_batch_size,
        encoding=args.encoding,
        engine=args.csv_engine,
        limit=args.limit,
    )

    try:
        first_chunk = next(chunk_iter)
    except StopIteration as exc:  # pragma: no cover - defensive
        raise ValueError("Input CSV yielded no data") from exc

    texts = first_chunk[args.text_column].fillna("")
    
    label_reader = pd.read_csv(
        args.csv_path,
        usecols=[args.label_column],
        encoding=args.encoding,
        engine=args.csv_engine,
        keep_default_na=False
    )
    races = label_reader[args.label_column].unique()
    label_map = {races[i]: i for i in range(len(races))}
    labels = first_chunk[args.label_column].map(label_map)

    inputs = prepare_inputs(texts.tolist(), encoder, args.instruction)
    first_embeddings = encoder._encode(inputs)
    if first_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {first_embeddings.shape}")

    embedding_dim = first_embeddings.shape[1]
    first_batch = embeddings_to_batch(first_embeddings.astype(np.float32), embedding_dim)

    label_chunks: List[np.ndarray] = [labels.to_numpy(dtype=np.float32)]

    with pa.OSFile(str(embedding_path), "wb") as sink:
        with ipc.new_file(sink, first_batch.schema) as writer:
            writer.write_batch(first_batch)
            written = len(first_embeddings)
            logging.info("Encoded chunk: %d/%d", written, total_rows)

            for chunk in chunk_iter:
                texts = chunk[args.text_column].fillna("")
                labels = chunk[args.label_column].transform(lambda x: label_map[x])
                inputs = prepare_inputs(texts.tolist(), encoder, args.instruction)
                batch_embeddings = encoder._encode(inputs).astype(np.float32)
                end = written + len(batch_embeddings)
                if end > total_rows:
                    raise ValueError(
                        f"Encountered more rows than expected: planned {total_rows}, attempting to write {end}"
                    )
                writer.write_batch(embeddings_to_batch(batch_embeddings, embedding_dim))
                label_chunks.append(labels.to_numpy(dtype=np.float32))
                written = end
                logging.info("Encoded chunk: %d/%d", written, total_rows)

        if written != total_rows:
            raise ValueError(f"Wrote {written} rows but expected {total_rows}")

    all_labels = np.concatenate(label_chunks).astype(np.float32)
    label_table = pa.table({"label": pa.array(all_labels, type=pa.float32())})
    feather.write_feather(label_table, str(labels_path))

    metadata = {
        "csv_path": str(args.csv_path.resolve()),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "model_alias": args.model,
        "encoder_class": encoder_cls.__name__,
        "max_input_length": args.max_input_length,
        "num_rows": total_rows,
        "embedding_dim": embedding_dim,
        "read_batch_size": args.read_batch_size,
        "instruction": args.instruction,
        "embedding_path": str(embedding_path.resolve()),
        "labels_path": str(labels_path.resolve()),
        "label_map": label_map
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logging.info("Finished writing embeddings to %s", embedding_path)
    logging.info("Finished writing labels to %s", labels_path)


if __name__ == "__main__":
    main()
