"""
Dynamic add example for LEANN using HNSW backend without recompute.

- Builds a base index from a directory of documents
- Incrementally adds new documents without recomputing stored embeddings

Defaults:
- Base data: /Users/yichuan/Desktop/code/LEANN/leann/data
- Incremental data: /Users/yichuan/Desktop/code/LEANN/leann/test_add
- Index path: <index_dir>/documents.leann

Usage examples:
  uv run python examples/dynamic_add_leann_no_recompute.py --build-base \
    --base-dir /Users/yichuan/Desktop/code/LEANN/leann/data \
    --index-dir ./test_doc_files

  uv run python examples/dynamic_add_leann_no_recompute.py --add-incremental \
    --add-dir /Users/yichuan/Desktop/code/LEANN/leann/test_add \
    --index-dir ./test_doc_files
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure we can import from the local packages and apps folders
ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "packages" / "leann-core" / "src"
HNSW_PKG_DIR = ROOT / "packages" / "leann-backend-hnsw"
APPS_DIR = ROOT / "apps"

# Prepend precise paths so the core module name `leann` resolves to leann-core
for p in [CORE_SRC, HNSW_PKG_DIR, APPS_DIR]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Defer non-stdlib imports until after sys.path setup within functions (avoid E402)


def _load_documents(data_dir: str, required_exts: Optional[list[str]] = None) -> list[Any]:
    from llama_index.core import SimpleDirectoryReader  # type: ignore

    reader_kwargs: dict[str, Any] = {"recursive": True, "encoding": "utf-8"}
    if required_exts:
        reader_kwargs["required_exts"] = required_exts
    documents = SimpleDirectoryReader(data_dir, **reader_kwargs).load_data(show_progress=True)
    return documents


def _ensure_index_dir(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)


def _index_files(index_path: Path) -> tuple[Path, Path, Path]:
    """Return (passages.jsonl, passages.idx, index.index) paths for a given index base path.

    Note: HNSWBackend writes the FAISS index using the stem (without .leann),
    i.e., for base 'documents.leann' the file is 'documents.index'. We prefer the
    existing file among candidates.
    """
    passages_file = index_path.parent / f"{index_path.name}.passages.jsonl"
    offsets_file = index_path.parent / f"{index_path.name}.passages.idx"
    candidate_name_index = index_path.parent / f"{index_path.name}.index"
    candidate_stem_index = index_path.parent / f"{index_path.stem}.index"
    index_file = candidate_stem_index if candidate_stem_index.exists() else candidate_name_index
    return passages_file, offsets_file, index_file


def _read_meta(index_path: Path) -> dict[str, Any]:
    meta_path = index_path.parent / f"{index_path.name}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _autodetect_index_base(index_dir: Path) -> Optional[Path]:
    """If exactly one *.leann.meta.json exists, return its base path (without .meta.json)."""
    candidates = list(index_dir.glob("*.leann.meta.json"))
    if len(candidates) == 1:
        meta = candidates[0]
        base = meta.with_suffix("")  # remove .json
        base = base.with_suffix("")  # remove .meta
        return base
    return None


def _load_offset_map(offsets_file: Path) -> dict[str, int]:
    if not offsets_file.exists():
        return {}
    with open(offsets_file, "rb") as f:
        return pickle.load(f)


def _next_numeric_id(existing_ids: list[str]) -> int:
    numeric_ids = [int(x) for x in existing_ids if x.isdigit()]
    if not numeric_ids:
        return 0
    return max(numeric_ids) + 1


def build_base_index(
    base_dir: str,
    index_dir: str,
    index_name: str,
    embedding_model: str,
    embedding_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    file_types: Optional[list[str]] = None,
    max_items: int = -1,
) -> str:
    print(f"Building base index from: {base_dir}")
    documents = _load_documents(base_dir, required_exts=file_types)
    if not documents:
        raise ValueError(f"No documents found in base_dir: {base_dir}")

    from chunking import create_text_chunks

    texts = create_text_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_ast_chunking=False,
    )
    if max_items > 0 and len(texts) > max_items:
        texts = texts[:max_items]
        print(f"Limiting to {max_items} chunks")

    index_dir_path = Path(index_dir)
    _ensure_index_dir(index_dir_path)
    index_path = index_dir_path / index_name

    print("Creating HNSW index with no-recompute (non-compact)...")
    from leann.api import LeannBuilder
    from leann.registry import register_project_directory

    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=embedding_model,
        embedding_mode=embedding_mode,
        is_recompute=False,
        is_compact=False,
    )
    for t in texts:
        builder.add_text(t)
    builder.build_index(str(index_path))

    # Register for discovery
    register_project_directory(Path.cwd())

    print(f"Base index built at: {index_path}")
    return str(index_path)


def add_incremental(
    add_dir: str,
    index_dir: str,
    index_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_mode: Optional[str] = None,
    chunk_size: int = 256,
    chunk_overlap: int = 128,
    file_types: Optional[list[str]] = None,
    max_items: int = -1,
) -> str:
    print(f"Adding incremental data from: {add_dir}")
    index_dir_path = Path(index_dir)
    index_path = index_dir_path / (index_name or "documents.leann")

    # If specified base doesn't exist, try to auto-detect an existing base
    meta = None
    try:
        meta = _read_meta(index_path)
    except FileNotFoundError:
        auto_base = _autodetect_index_base(index_dir_path)
        if auto_base is not None:
            print(f"Auto-detected index base: {auto_base.name}")
            index_path = auto_base
            meta = _read_meta(index_path)
        else:
            raise FileNotFoundError(
                f"No index metadata found for base '{index_path.name}'. Build base first with --build-base "
                f"or provide --index-name to match an existing index (e.g., 'test_doc_files.leann')."
            )

    passages_file, offsets_file, faiss_index_file = _index_files(index_path)

    if meta.get("backend_name") != "hnsw":
        raise RuntimeError("Incremental add is currently supported only for HNSW backend")
    if meta.get("is_compact", True):
        raise RuntimeError(
            "Index is compact. Rebuild base with --no-recompute and --no-compact for incremental add."
        )

    # Ensure the vector index exists before appending passages
    if not faiss_index_file.exists():
        raise FileNotFoundError(
            f"Vector index file missing: {faiss_index_file}. Build base first (use --build-base)."
        )

    # Resolve embedding config from meta if not provided
    emb_model = embedding_model or meta.get("embedding_model", "facebook/contriever")
    emb_mode = embedding_mode or meta.get("embedding_mode", "sentence-transformers")

    documents = _load_documents(add_dir, required_exts=file_types)
    if not documents:
        raise ValueError(f"No documents found in add_dir: {add_dir}")

    from chunking import create_text_chunks

    new_texts = create_text_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_ast_chunking=False,
    )
    if max_items > 0 and len(new_texts) > max_items:
        new_texts = new_texts[:max_items]
        print(f"Limiting to {max_items} chunks (incremental)")

    if not new_texts:
        print("No new chunks to add.")
        return str(index_path)

    # Load and extend passages + offsets
    offset_map = _load_offset_map(offsets_file)
    start_id_int = _next_numeric_id(list(offset_map.keys()))
    next_id = start_id_int

    # Append to passages.jsonl and collect offsets
    print("Appending passages and updating offsets...")
    with open(passages_file, "a", encoding="utf-8") as f:
        for text in new_texts:
            offset = f.tell()
            str_id = str(next_id)
            json.dump({"id": str_id, "text": text, "metadata": {}}, f, ensure_ascii=False)
            f.write("\n")
            offset_map[str_id] = offset
            next_id += 1

    with open(offsets_file, "wb") as f:
        pickle.dump(offset_map, f)

    # Compute embeddings for new texts
    print("Computing embeddings for incremental chunks...")
    from leann.api import compute_embeddings

    embeddings = compute_embeddings(
        new_texts,
        model_name=emb_model,
        mode=emb_mode,
        use_server=False,
        is_build=True,
    )

    # Load FAISS HNSW index and add vectors, then write back
    print("Loading HNSW index and appending vectors...")
    try:
        from leann_backend_hnsw import faiss as hnsw_faiss  # type: ignore
    except Exception as e:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Failed to import leann_backend_hnsw.faiss. Ensure backend is built/installed."
        ) from e

    # Read existing index
    # Read existing index (basic read is sufficient for appending)
    index = hnsw_faiss.read_index(str(faiss_index_file), hnsw_faiss.IO_FLAG_MMAP)

    # Normalize for cosine if needed
    distance_metric = meta.get("backend_kwargs", {}).get("distance_metric", "mips").lower()
    if distance_metric == "cosine":
        import numpy as _np

        norms = _np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

    # Ensure dtype float32 and contiguous
    import numpy as _np

    if embeddings.dtype != _np.float32:
        embeddings = embeddings.astype(_np.float32)
    if not embeddings.flags.c_contiguous:
        embeddings = _np.ascontiguousarray(embeddings, dtype=_np.float32)

    # Append using FAISS-style signature (n, swig_ptr(x)); fall back to Python wrapper if needed
    try:
        index.add(embeddings.shape[0], hnsw_faiss.swig_ptr(embeddings))
    except TypeError:
        # Some builds expose a Python-friendly signature: index.add(np.ndarray)
        index.add(embeddings)
    hnsw_faiss.write_index(index, str(faiss_index_file))

    print(f"Incremental add completed. Index updated at: {index_path}")
    return str(index_path)


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic add to LEANN HNSW index without recompute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--build-base", action="store_true", help="Build base index")
    parser.add_argument("--add-incremental", action="store_true", help="Add incremental data")

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/Users/yichuan/Desktop/code/LEANN/leann/data",
        help="Base data directory",
    )
    parser.add_argument(
        "--add-dir",
        type=str,
        default="/Users/yichuan/Desktop/code/LEANN/leann/test_add",
        help="Incremental data directory",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./test_doc_files",
        help="Directory containing the index",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="documents.leann",
        help=(
            "Index base file name. If you built via document_rag.py, use 'test_doc_files.leann'. "
            "Default: documents.leann"
        ),
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="facebook/contriever",
        help="Embedding model name",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "mlx", "ollama"],
        help="Embedding backend mode",
    )

    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument("--file-types", nargs="+", default=None)
    parser.add_argument("--max-items", type=int, default=-1)

    args = parser.parse_args()

    if not args.build_base and not args.add_incremental:
        print("Nothing to do. Use --build-base and/or --add-incremental.")
        return

    index_path_str: Optional[str] = None

    if args.build_base:
        index_path_str = build_base_index(
            base_dir=args.base_dir,
            index_dir=args.index_dir,
            index_name=args.index_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            file_types=args.file_types,
            max_items=args.max_items,
        )

    if args.add_incremental:
        index_path_str = add_incremental(
            add_dir=args.add_dir,
            index_dir=args.index_dir,
            index_name=args.index_name,
            embedding_model=args.embedding_model,
            embedding_mode=args.embedding_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            file_types=args.file_types,
            max_items=args.max_items,
        )

    # Optional: quick test query using searcher
    if index_path_str:
        try:
            from leann.api import LeannSearcher

            searcher = LeannSearcher(index_path_str)
            query = "what is LEANN?"
            if args.add_incremental:
                query = "what is the multi vector search and how it works?"
            results = searcher.search(query, top_k=5, recompute_embeddings=False)
            if results:
                print(f"Sample result: {results[0].text[:80]}...")
        except Exception:
            pass


if __name__ == "__main__":
    main()
