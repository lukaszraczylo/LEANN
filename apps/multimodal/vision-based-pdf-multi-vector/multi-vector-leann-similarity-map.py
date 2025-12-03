## Jupyter-style notebook script
# %%
# uv pip install matplotlib qwen_vl_utils
import os
from typing import Any, Optional

from PIL import Image
from tqdm import tqdm


from leann_multi_vector import (  # utility functions/classes
    _ensure_repo_paths_importable,
    _load_images_from_dir,
    _maybe_convert_pdf_to_images,
    _load_colvision,
    _embed_images,
    _embed_queries,
    _build_index,
    _load_retriever_if_index_exists,
    _generate_similarity_map,
    QwenVL,
)

_ensure_repo_paths_importable(__file__)

# %%
# Config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
QUERY = "The paper talk about the latent video generative model and data curation in the related work part?"
MODEL: str = "colqwen2"  # "colpali" or "colqwen2"

# Data source: set to True to use the Hugging Face dataset example (recommended)
USE_HF_DATASET: bool = True
DATASET_NAME: str = "weaviate/arXiv-AI-papers-multi-vector"
DATASET_SPLIT: str = "train"
MAX_DOCS: Optional[int] = None  # limit number of pages to index; None = all

# Local pages (used when USE_HF_DATASET == False)
PDF: Optional[str] = None  # e.g., "./pdfs/2004.12832v2.pdf"
PAGES_DIR: str = "./pages"

# Index + retrieval settings
INDEX_PATH: str = "./indexes/colvision.leann"
TOPK: int = 3
FIRST_STAGE_K: int = 500
REBUILD_INDEX: bool = False

# Artifacts
SAVE_TOP_IMAGE: Optional[str] = "./figures/retrieved_page.png"
SIMILARITY_MAP: bool = True
SIM_TOKEN_IDX: int = 13  # -1 means auto-select the most salient token
SIM_OUTPUT: str = "./figures/similarity_map.png"
ANSWER: bool = True
MAX_NEW_TOKENS: int = 1024


# %%

# Step 1: Check if we can skip data loading (index already exists)
retriever: Optional[Any] = None
need_to_build_index = REBUILD_INDEX

if not REBUILD_INDEX:
    retriever = _load_retriever_if_index_exists(INDEX_PATH)
    if retriever is not None:
        print(f"✓ Index loaded from {INDEX_PATH}")
        print(f"✓ Images available at: {retriever._images_dir_path()}")
        need_to_build_index = False
    else:
        print(f"Index not found, will build new index")
        need_to_build_index = True

# Step 2: Load data only if we need to build the index
if need_to_build_index:
    print("Loading dataset...")
    if USE_HF_DATASET:
        from datasets import load_dataset

        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        N = len(dataset) if MAX_DOCS is None else min(MAX_DOCS, len(dataset))
        filepaths: list[str] = []
        images: list[Image.Image] = []
        for i in tqdm(range(N), desc="Loading dataset", total=N):
            p = dataset[i]
            # Compose a descriptive identifier for printing later
            identifier = f"arXiv:{p['paper_arxiv_id']}|title:{p['paper_title']}|page:{int(p['page_number'])}|id:{p['page_id']}"
            filepaths.append(identifier)
            images.append(p["page_image"])  # PIL Image
    else:
        _maybe_convert_pdf_to_images(PDF, PAGES_DIR)
        filepaths, images = _load_images_from_dir(PAGES_DIR)
        if not images:
            raise RuntimeError(
                f"No images found in {PAGES_DIR}. Provide PDF path in PDF variable or ensure images exist."
            )
    print(f"Loaded {len(images)} images")
else:
    print("Skipping dataset loading (using existing index)")
    filepaths = []  # Not needed when using existing index
    images = []  # Not needed when using existing index


# %%
# Step 3: Load model and processor (only if we need to build index or perform search)
model_name, model, processor, device_str, device, dtype = _load_colvision(MODEL)
print(f"Using model={model_name}, device={device_str}, dtype={dtype}")


# %%

# %%
# Step 4: Build index if needed
if need_to_build_index and retriever is None:
    print("Building index...")
    doc_vecs = _embed_images(model, processor, images)
    retriever = _build_index(INDEX_PATH, doc_vecs, filepaths, images)
    print(f"✓ Index built and images saved to: {retriever._images_dir_path()}")
    # Clear memory
    del images, filepaths, doc_vecs

# Note: Images are now stored in the index, retriever will load them on-demand from disk


# %%
# Step 5: Embed query and search
q_vec = _embed_queries(model, processor, [QUERY])[0]
results = retriever.search(q_vec.float().numpy(), topk=TOPK)
if not results:
    print("No results found.")
else:
    print(f'Top {len(results)} results for query: "{QUERY}"')
    top_images: list[Image.Image] = []
    for rank, (score, doc_id) in enumerate(results, start=1):
        # Retrieve image from index instead of memory
        image = retriever.get_image(doc_id)
        if image is None:
            print(f"Warning: Could not retrieve image for doc_id {doc_id}")
            continue

        metadata = retriever.get_metadata(doc_id)
        path = metadata.get("filepath", "unknown") if metadata else "unknown"
        # For HF dataset, path is a descriptive identifier, not a real file path
        print(f"{rank}) MaxSim: {score:.4f}, Page: {path}")
        top_images.append(image)

    if SAVE_TOP_IMAGE:
        from pathlib import Path as _Path

        base = _Path(SAVE_TOP_IMAGE)
        base.parent.mkdir(parents=True, exist_ok=True)
        for rank, img in enumerate(top_images[:TOPK], start=1):
            if base.suffix:
                out_path = base.parent / f"{base.stem}_rank{rank}{base.suffix}"
            else:
                out_path = base / f"retrieved_page_rank{rank}.png"
            img.save(str(out_path))
            # Print the retrieval score (document-level MaxSim) alongside the saved path
            try:
                score, _doc_id = results[rank - 1]
                print(f"Saved retrieved page (rank {rank}) [MaxSim={score:.4f}] to: {out_path}")
            except Exception:
                print(f"Saved retrieved page (rank {rank}) to: {out_path}")

## TODO stange results of second page of DeepSeek-V2 rather than the first page

# %%
# Step 6: Similarity maps for top-K results
if results and SIMILARITY_MAP:
    token_idx = None if SIM_TOKEN_IDX < 0 else int(SIM_TOKEN_IDX)
    from pathlib import Path as _Path

    output_base = _Path(SIM_OUTPUT) if SIM_OUTPUT else None
    for rank, img in enumerate(top_images[:TOPK], start=1):
        if output_base:
            if output_base.suffix:
                out_dir = output_base.parent
                out_name = f"{output_base.stem}_rank{rank}{output_base.suffix}"
                out_path = str(out_dir / out_name)
            else:
                out_dir = output_base
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = str(out_dir / f"similarity_map_rank{rank}.png")
        else:
            out_path = None
        chosen_idx, max_sim = _generate_similarity_map(
            model=model,
            processor=processor,
            image=img,
            query=QUERY,
            token_idx=token_idx,
            output_path=out_path,
        )
        if out_path:
            print(
                f"Saved similarity map for rank {rank}, token #{chosen_idx} (max={max_sim:.2f}) to: {out_path}"
            )
        else:
            print(
                f"Computed similarity map for rank {rank}, token #{chosen_idx} (max={max_sim:.2f})"
            )


# %%
# Step 7: Optional answer generation
if results and ANSWER:
    qwen = QwenVL(device=device_str)
    response = qwen.answer(QUERY, top_images[:TOPK], max_new_tokens=MAX_NEW_TOKENS)
    print("\nAnswer:")
    print(response)
