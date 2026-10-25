"""Qwen3 decomposition and extraction evidence generator."""
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

FINAL_DIR = Path(__file__).resolve().parent
SHARED_DIR = FINAL_DIR / 'common'
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

ENABLE_THINKING = True
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95


class HFBackend:
    def __init__(self, model, dtype="auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        if dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map="auto", attn_implementation="sdpa"
        ).eval()

    def generate(self, prompt, images, n, temperature, top_p, top_k, max_tokens, system_prompt=None, image_labels=None):
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=ENABLE_THINKING)
        inputs = self.tokenizer([text], return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        context_limit = self.model.config.max_position_embeddings
        output_budget = min(max_tokens, context_limit - input_length)
        if output_budget <= 0:
            raise ValueError(f"Input has {input_length} tokens; model context limit is {context_limit}")
        if ENABLE_THINKING and temperature <= 0:
            temperature, top_p, top_k = DEFAULT_TEMPERATURE, DEFAULT_TOP_P, 20
        opts = {"max_new_tokens": output_budget, "do_sample": temperature > 0}
        if temperature > 0:
            opts.update(temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0)

        result = []
        think_end = self.tokenizer.convert_tokens_to_ids("</think>")
        with self.torch.inference_mode():
            for _ in range(n):
                out = self.model.generate(**inputs, **opts)
                output_ids = out[0, input_length:].tolist()
                eos_ids = self.model.generation_config.eos_token_id
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                ended = bool(output_ids and output_ids[-1] in (eos_ids or []))
                boundary = max((i for i, token in enumerate(output_ids) if token == think_end), default=-1)
                complete = boundary >= 0 or not ENABLE_THINKING
                answer_ids = output_ids[boundary + 1:] if boundary >= 0 else ([] if ENABLE_THINKING else output_ids)
                reasoning_ids = output_ids[:boundary] if boundary >= 0 else (output_ids if ENABLE_THINKING else [])
                result.append({
                    "text": self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
                    "reasoning": self.tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip(),
                    "raw_output": self.tokenizer.decode(output_ids, skip_special_tokens=False),
                    "thinking_complete": complete,
                    "input_tokens": input_length, "output_tokens": len(output_ids),
                    "output_budget": output_budget,
                    "finish_reason": "length" if len(output_ids) >= output_budget and not ended else "stop",
                })
        return result


def load_records(path):
    records = {}
    if path and Path(path).is_file():
        with Path(path).open(encoding='utf-8-sig') as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get('status') == 'ok' and 'id' in record:
                    records[str(record['id'])] = record
    return records



def append_record(path, record):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open('rb') as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) != b'\n'
        if needs_newline:
            with path.open('ab') as handle:
                handle.write(b'\n')
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        handle.flush()


def compact_record(record):
    return {
        'id': record['id'], 'status': record['status'], 'model': record['model'],
        'subqueries': record['subqueries'],
        'extractions': [{key: item[key] for key in ('query', 'answer', 'status')}
                        for item in record['extractions']],
    }


def run(rows, backend, settings, output_dir, pipeline):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = failed = unresolved = 0
    for i, row in enumerate(rows):
        sample_id = pipeline.normalize_sample(row)['id']
        record = dict(id=sample_id, model=settings.model, decomposition_model=settings.model,
                      extraction_model=settings.model, temperature=settings.temperature,
                      top_p=settings.top_p, top_k=settings.top_k, max_tokens=settings.max_tokens)
        try:
            pipeline.solve(row, backend, None, settings, record, evidence_only=True)
        except Exception as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {exc}')
            failed += 1
            print(f'[{i+1}/{len(rows)}] {sample_id}: {record["error"]}', flush=True)
            if type(exc).__name__ == 'OutOfMemoryError':
                raise
            continue
        record['status'] = 'ok'
        append_record(output_dir / getattr(settings, 'evidence_filename', 'dec_ex.jsonl'), compact_record(record))
        completed += 1
        count = sum(item.get('status') != 'ok' for item in record['extractions'])
        unresolved += count
        print(f'[{i+1}/{len(rows)}] {sample_id}: ok; queries={len(record["subqueries"])}; unresolved={count}', flush=True)
    return dict(completed=completed, failed=failed, unresolved_extractions=unresolved)


if __name__ == "__main__":
    import pipeline

    SPLIT = 'public'
    if SPLIT not in ('train', 'public', 'private'):
        raise ValueError('Unknown SPLIT')
    DATA_STEM = 'train' if SPLIT == 'train' else SPLIT + '_test'
    INPUT_FILE = FINAL_DIR / 'data' / f'{DATA_STEM}.json'
    IMAGE_ROOT = FINAL_DIR / 'data' / f'{DATA_STEM}_images'
    CHART_CAPTION_FILE = FINAL_DIR / 'data' / 'caption' / f'{DATA_STEM}_caption.jsonl'
    MODEL_DIR = 'Qwen/Qwen3-8B'
    SHARD_ID, N_SHARDS = 0, 1
    OUTPUT_DIR = FINAL_DIR / 'outputs' / f'dec_ex_{SPLIT}' / f'shard_{SHARD_ID:02d}'
    LIMIT = None
    TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20
    MAX_NEW_TOKENS = 16384 * 2
    DTYPE = 'auto'

    rows = pipeline.read_rows(INPUT_FILE)
    if N_SHARDS < 1 or not 0 <= SHARD_ID < N_SHARDS:
        raise ValueError('Require N_SHARDS >= 1 and 0 <= SHARD_ID < N_SHARDS')
    total_rows = len(rows)
    start = total_rows * SHARD_ID // N_SHARDS
    end = total_rows * (SHARD_ID + 1) // N_SHARDS
    rows = rows[start:end]
    if LIMIT is not None:
        rows = rows[:LIMIT]

    chart_captions = pipeline.load_chart_captions(CHART_CAPTION_FILE)
    image_files = {Path(image).name for row in rows for image in pipeline.prepare_sample(row, IMAGE_ROOT)[1]}
    missing = sorted(image_files - chart_captions.keys())
    print(f'Selected samples: {len(rows)}/{total_rows} | rows[{start}:{end}]')
    print(f'Unique images: {len(image_files)} | Loaded captions: {len(chart_captions)} | Missing: {len(missing)}')

    settings = SimpleNamespace(model=str(MODEL_DIR), image_root=str(IMAGE_ROOT),
                               chart_captions=chart_captions, temperature=TEMPERATURE,
                               top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_NEW_TOKENS,
                               evidence_filename=f'dec_ex_{SPLIT}.jsonl')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    compact_file = OUTPUT_DIR / settings.evidence_filename
    compact_file.write_text('', encoding='utf-8')
    pending = rows
    print(f'Samples to generate: {len(pending)}')

    summary = dict(completed=0, failed=0, unresolved_extractions=0)
    if pending:
        backend = HFBackend(str(MODEL_DIR), dtype=DTYPE)
        try:
            summary = run(pending, backend, settings, OUTPUT_DIR, pipeline)
        finally:
            del backend
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(summary)

    compact_file = OUTPUT_DIR / settings.evidence_filename
    print('Shard evidence:', compact_file)
    merged = {}
    for shard_file in sorted(OUTPUT_DIR.parent.glob(f'shard_*/dec_ex_{SPLIT}.jsonl')):
        for key, record in load_records(shard_file).items():
            if key in merged and merged[key] != record:
                raise ValueError('Conflicting shard evidence: ' + key)
            merged[key] = record
    all_rows = pipeline.read_rows(INPUT_FILE)
    all_ids = [str(pipeline.normalize_sample(row)['id']) for row in all_rows]
    if set(merged) == set(all_ids):
        destination = FINAL_DIR / 'data' / 'decompose_extraction_evidence' / settings.evidence_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.jsonl.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            for key in all_ids:
                handle.write(json.dumps(merged[key], ensure_ascii=False) + '\n')
        temporary.replace(destination)
        print('Evidence for augment/infer:', destination)
    else:
        print(f'Combined evidence incomplete: {len(set(merged) & set(all_ids))}/{len(all_ids)}; destination unchanged')
    