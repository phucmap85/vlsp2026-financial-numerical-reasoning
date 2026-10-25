import json
import re
import sys
from pathlib import Path

import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

FINAL_DIR = Path(__file__).resolve().parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

SYSTEM_PROMPT = """You are a chart image transcription module.

Report only information that is directly visible in the image.
An image may contain one or multiple charts.

Rules:
- Copy visible text as exactly as possible.
- Preserve Vietnamese accents, punctuation, symbols, dates, and abbreviations.
- Do not translate, summarize, paraphrase, or create new labels.
- Do not infer missing information.
- Do not estimate values from bar heights, line positions, areas, or shapes.
- Copy numerical values only when they are explicitly printed.
- Do not confuse X-axis values with Y-axis ticks.
- For pie/donut charts, axes are usually absent; copy visible legend and slice labels.
- If one plot combines multiple chart types, use mixed(type1,type2).
- If information is not clearly visible, write none.

Return only the requested format."""

USER_PROMPT = """Inspect this image.

First identify whether it contains one chart or multiple visually distinct charts.

Return:

FIGURE_TITLE: <overall visible title for the whole image or none>
CHART_COUNT: <number of visually distinct charts>

For each chart, repeat this block:

[CHART 1]
TYPE: <bar | line | pie | donut | scatter | area | mixed(...) | other>
TITLE: <title specific to this chart or none>
LEGEND: <legend item 1> | <legend item 2> | ... | none
X_AXIS_LABEL: <exact visible label or none>
X_AXIS_VALUES: <all visible values/categories from left to right separated by | or none>
X_AXIS_GROUPS: <higher-level visible groups from left to right separated by | or none>
Y_AXIS_LABEL: <exact visible label or none>
Y_AXIS_TICKS: <all visible ticks from bottom to top separated by | or none>
RIGHT_Y_AXIS_LABEL: <exact visible label or none>
RIGHT_Y_AXIS_TICKS: <all visible ticks from bottom to top separated by | or none>
UNIT: <explicitly visible unit or none>
DATA_LABELS: <values/labels printed directly on bars, points, slices, etc. separated by | or none>

If there are more charts, continue with:

[CHART 2]
...

Return nothing else."""

SCHEMA_KEYS = [
    "TYPE", "TITLE", "FIELDS", "SELECTOR_1_NAME", "SELECTOR_1_VALUES",
    "SELECTOR_1_ORDERED", "SELECTOR_2_NAME", "SELECTOR_2_VALUES", "UNIT", "UNCERTAIN"
]


def load_dataframe(path):
    import pandas as pd
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".json":
        with open(path, 'r', encoding='utf-8') as f:
            return pd.json_normalize(json.load(f))
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file: {path}")


def detect_image_columns(df):
    pattern = re.compile(r"(?:^|[.\s])Image\s*\d+$", flags=re.IGNORECASE)
    return [col for col in df.columns if pattern.search(str(col))]


def load_image(path, max_long_edge=None):
    from PIL import Image, ImageOps
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max_long_edge is not None:
        width, height = image.size
        longest = max(width, height)
        if longest > max_long_edge:
            scale = max_long_edge / longest
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    return image


def build_conversation(image):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": USER_PROMPT}
        ]}
    ]


def remove_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.replace("```text", "").replace("```", "").strip()


def parse_schema(text):
    text = remove_thinking(text)
    pattern = re.compile(r"(?m)^(" + "|".join(map(re.escape, SCHEMA_KEYS)) + r")\s*:\s*")
    matches = list(pattern.finditer(text))
    parsed = {}
    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parsed[key] = re.sub(r"\s*\n\s*", " ", text[start:end].strip())
    return parsed


def canonical_schema_text(parsed):
    def get(k):
        v = parsed.get(k)
        return str(v).strip() if v is not None and str(v).strip() else "none"
    return "\n".join(f"{k}: {get(k)}" for k in SCHEMA_KEYS)


def empty_record(meta, status, error=None, thinking_enabled=False, raw_output=None):
    return {
        **meta, "status": status, "error": error, "thinking_enabled": thinking_enabled,
        "raw_output": raw_output, "schema_text": None, "chart_type": None, "title": None,
        "fields": None, "selector_1_name": None, "selector_1_values": None,
        "selector_1_ordered": None, "selector_2_name": None, "selector_2_values": None,
        "unit": None, "uncertain": None
    }


if __name__ == "__main__":
    import pandas as pd
    from tqdm.auto import tqdm
    from vllm import LLM, SamplingParams

    SPLIT = 'private'
    if SPLIT not in ('train', 'public', 'private'):
        raise ValueError('Unknown SPLIT')
    DATA_STEM = 'train' if SPLIT == 'train' else SPLIT + '_test'
    MODEL_PATH = 'Qwen/Qwen3.5-4B'
    DTYPE = "float16"
    TENSOR_PARALLEL_SIZE = 1
    GPU_MEMORY_UTILIZATION = 0.90
    MAX_MODEL_LEN = 4096
    MAX_NUM_SEQS = 8
    ENABLE_THINKING = False
    MAX_OUTPUT_TOKENS = 2048 if ENABLE_THINKING else 1024
    BATCH_SIZE = 16
    TEMPERATURE = 0.0
    TOP_P = 1.0
    SEED = 42
    MAX_LONG_EDGE = None
    TEST_MODE = False
    TEST_N_IMAGES = 10
    RESUME = True
    SAVE_RAW_OUTPUT = True

    TRAIN_PATH = FINAL_DIR / 'data' / f'{DATA_STEM}.json'
    IMAGE_DIR = FINAL_DIR / 'data' / f'{DATA_STEM}_images'
    OUTPUT_DIR = FINAL_DIR / 'outputs' / 'caption' / SPLIT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PARQUET = OUTPUT_DIR / f'{DATA_STEM}_chart_perception.parquet'
    OUTPUT_CSV = OUTPUT_DIR / f'{DATA_STEM}_chart_perception.csv'
    CAPTION_DIR = FINAL_DIR / 'data' / 'caption'
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RAW_JSONL = CAPTION_DIR / f'{DATA_STEM}_caption.jsonl'
    train_df = load_dataframe(TRAIN_PATH)
    image_cols = detect_image_columns(train_df)
    referenced_files = sorted({
        Path(str(val).strip()).name for col in image_cols for val in train_df[col]
        if pd.notna(val) and Path(str(val).strip()).name
    })

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    image_index = {p.name: str(p) for p in IMAGE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    jobs = [{"image_file": fn, "image_path": image_index.get(fn)} for fn in referenced_files if image_index.get(fn)]
    jobs_df = pd.DataFrame(jobs)
    print(f"Total images referenced: {len(referenced_files)}, found: {len(jobs_df)}")

    existing_df = pd.read_parquet(OUTPUT_PARQUET) if (RESUME and OUTPUT_PARQUET.exists()) else pd.DataFrame()
    processed_files = set(existing_df["image_file"].dropna().astype(str)) if len(existing_df) else set()
    pending_df = jobs_df[~jobs_df["image_file"].isin(processed_files)].reset_index(drop=True)
    if TEST_MODE:
        pending_df = pending_df.head(min(TEST_N_IMAGES, len(pending_df)))
    print(f"Already processed: {len(processed_files)} | Pending: {len(pending_df)}")

    llm = LLM(model=MODEL_PATH, trust_remote_code=True, dtype=DTYPE,
              tensor_parallel_size=TENSOR_PARALLEL_SIZE, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
              max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS, limit_mm_per_prompt={"image": 1})
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_OUTPUT_TOKENS, seed=SEED)

    new_results = []
    for start in tqdm(range(0, len(pending_df), BATCH_SIZE), desc="Chart perception"):
        batch = pending_df.iloc[start:start + BATCH_SIZE]
        conversations, metadata = [], []
        for _, row in batch.iterrows():
            meta = {"image_file": row["image_file"], "image_path": row["image_path"]}
            try:
                img = load_image(row["image_path"], max_long_edge=MAX_LONG_EDGE)
                conversations.append(build_conversation(img))
                metadata.append(meta)
            except Exception as e:
                new_results.append(empty_record(meta, "image_error", repr(e), ENABLE_THINKING))

        if not conversations:
            continue
        try:
            outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False,
                               chat_template_kwargs={"enable_thinking": ENABLE_THINKING})
        except Exception as e:
            for meta in metadata:
                new_results.append(empty_record(meta, "inference_error", repr(e), ENABLE_THINKING))
            continue

        for meta, output in zip(metadata, outputs):
            try:
                raw_text = output.outputs[0].text.strip()
                parsed = parse_schema(raw_text)
                schema_text = canonical_schema_text(parsed)
                result = {
                    **meta, "status": "ok", "error": None, "thinking_enabled": ENABLE_THINKING,
                    "raw_output": raw_text, "schema_text": schema_text,
                    "chart_type": parsed.get("TYPE"), "title": parsed.get("TITLE"), "fields": parsed.get("FIELDS"),
                    "selector_1_name": parsed.get("SELECTOR_1_NAME"), "selector_1_values": parsed.get("SELECTOR_1_VALUES"),
                    "selector_1_ordered": parsed.get("SELECTOR_1_ORDERED"), "selector_2_name": parsed.get("SELECTOR_2_NAME"),
                    "selector_2_values": parsed.get("SELECTOR_2_VALUES"), "unit": parsed.get("UNIT"),
                    "uncertain": parsed.get("UNCERTAIN")
                }
                new_results.append(result)
                if SAVE_RAW_OUTPUT:
                    raw_record = {"image_file": meta["image_file"], "image_path": meta["image_path"],
                                  "thinking_enabled": ENABLE_THINKING, "raw_output": raw_text}
                    with open(OUTPUT_RAW_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            except Exception as e:
                new_results.append(empty_record(meta, "parse_error", repr(e), ENABLE_THINKING,
                                                raw_output=output.outputs[0].text.strip()))

        new_df = pd.DataFrame(new_results)
        save_df = pd.concat([existing_df, new_df], ignore_index=True) if len(existing_df) else new_df.copy()
        save_df = save_df.drop_duplicates(subset=["image_file"], keep="last").reset_index(drop=True)
        save_df.to_parquet(OUTPUT_PARQUET, index=False)
        save_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Output parquet:", OUTPUT_PARQUET)
    print("Output CSV:", OUTPUT_CSV)