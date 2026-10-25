import gc
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_flow(code):
    directory = str(Path(code).resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import pipeline, planning
    return pipeline, planning


def shard_rows(rows, shard_id, n_shards):
    if not 0 <= shard_id < n_shards:
        raise ValueError('SHARD_ID must be in 0..N_SHARDS-1')
    start, end = len(rows) * shard_id // n_shards, len(rows) * (shard_id + 1) // n_shards
    return start, rows[start:end]


def read_evidence(path, read_rows):
    records = {}
    for row in read_rows(path):
        qid = str(row.get('qid') or row.get('id') or '')
        if not qid:
            raise ValueError('Evidence record without QID')
        if qid in records and records[qid] != row:
            raise ValueError('Conflicting evidence records: ' + qid)
        records[qid] = row
    return records


def make_sample(raw, evidence, captions, hits, pipeline, image_index):
    import re
    from multimodal_dataset import build_context, make_input_messages, image_map, IMAGE_MARKER
    from prompt_contract import planner_user, render_inline_context, select_examples
    norm = pipeline.normalize_sample(raw)
    queries = evidence.get('subqueries')
    if (not isinstance(queries, list) or not 1 <= len(queries) <= 8
            or not all(isinstance(q, str) and q.strip() for q in queries)):
        raise ValueError('Missing/invalid cached subqueries: ' + str(norm['id']))
    extractions = evidence.get('extractions')
    if (not isinstance(extractions, list) or len(extractions) != len(queries)
            or any(not isinstance(e, dict) or e.get('query') != q
                   or not isinstance(e.get('answer'), str)
                   for q, e in zip(queries, extractions))):
        raise ValueError('Cached extractions do not match queries: ' + str(norm['id']))
    context = build_context(raw, captions)
    missing = set(context['image_labels'].values()) - set(context.get('chart_perception', {}))
    if missing:
        raise ValueError('Missing captions: ' + str(sorted(missing)))
    examples, audit = select_examples(hits, norm['question'], norm['id'],
                                     pipeline.submission_to_planner, top_k=5)
    messages = [{'role': 'system', 'content': pipeline.PLANNING_SYSTEM},
                {'role': 'user', 'content': planner_user(render_inline_context(context),
                                                       norm['question'], extractions, examples)}]
    labels = image_map(raw)
    unmapped = sorted({'Image ' + str(int(re.search(r'\d+', match[1])[0]))
                      for match in IMAGE_MARKER.finditer(messages[1]['content'])} - set(labels))
    messages = make_input_messages(messages, raw, image_index, allow_unmapped_markers=True)
    return {'qid': str(norm['id']), 'input': messages, 'context': context,
            'retrieval_audit': audit, 'image_markers_without_files': unmapped,
            'subqueries': queries,
            'decomposition_model': evidence.get('decomposition_model', evidence.get('model')),
            'decomposition_fallback': evidence.get('decomposition_fallback'),
            'extractions': extractions, 'extraction_model': evidence.get('extraction_model')}


def prepare(config):
    c = config
    pipeline, planning = load_flow(c['CODE_DIR'])
    rows = planning.read_rows(c['INPUT_FILE'])
    ids = [str(pipeline.normalize_sample(r)['id']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate input QIDs')
    start, selected = shard_rows(rows, c['SHARD_ID'], c['N_SHARDS'])
    if c['LIMIT'] is not None:
        selected = selected[:c['LIMIT']]
    out = Path(c['OUTPUT_ROOT']) / f"shard_{c['SHARD_ID']:02d}"
    out.mkdir(parents=True, exist_ok=True)
    captions = pipeline.load_chart_captions(c['CHART_CAPTION_FILE'])
    evidence = read_evidence(c['EVIDENCE_FILE'], planning.read_rows)
    from multimodal_dataset import index_images
    image_index = index_images(c['IMAGE_ROOT'])
    train = []
    for raw in planning.read_rows(c['TRAIN_FILE']):
        row = pipeline.normalize_sample(raw)
        train.append({'id': str(row['id']), 'qa': {'question': row['question'],
            'program': raw.get('qa', {}).get('program') or raw.get('program', '')}})
    train_path = out / 'retrieval_train.json'
    train_path.write_text(json.dumps(train, ensure_ascii=False), encoding='utf-8')
    from hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(str(train_path), str(c['BGE_MODEL_DIR']),
                                str(out / 'bge_m3_train_embeddings.pt'))
    index_by_id = {r['id']: i for i, r in enumerate(train)}
    samples = []
    for offset, raw in enumerate(selected):
        norm = pipeline.normalize_sample(raw)
        qid = str(norm['id'])
        hits = retriever.search(norm['question'], top_k=10, exclude_index=index_by_id.get(qid))
        hits = [dict(h, qid=train[h['index']]['id'], program=h['answer']) for h in hits]
        sample = make_sample(raw, evidence.get(qid, {}), captions, hits, pipeline, image_index)
        sample['source_row_index'] = start + offset
        # Read private gold only for optional reporting, after model input is finalized.
        sample['gold'] = raw.get('qa', {}).get('program') or raw.get('program')
        samples.append(sample)
    del retriever
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    (out / 'prepared_inputs.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n'
                                                       for s in samples), encoding='utf-8')
    print(f"Total={len(rows)}, shard={c['SHARD_ID']}/{c['N_SHARDS']}, samples={len(samples)}")
    return pipeline, planning, samples, out


def adapter_info(path):
    path = Path(path)
    config = json.loads((path / 'adapter_config.json').read_text(encoding='utf-8'))
    weights = sorted(path.glob('adapter_model.*'))
    if not weights:
        raise FileNotFoundError('No adapter_model weights: ' + str(path))
    hashes = {}
    for weight in weights:
        h = hashlib.sha256()
        with weight.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        hashes[weight.name] = h.hexdigest()
    return config, hashes


def decode_candidate(tokenizer, ids, pipeline, thinking, finish_reason, input_tokens, budget):
    raw = tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text, reasoning, complete = pipeline.split_thinking_text(raw, thinking)
    for token in tokenizer.all_special_tokens:
        if token not in ('<think>', '</think>'):
            text, reasoning = text.replace(token, ''), reasoning.replace(token, '')
    return {'text': text.strip(), 'reasoning': reasoning.strip(), 'raw_output': raw,
            'thinking_complete': complete, 'finish_reason': finish_reason,
            'input_tokens': input_tokens, 'output_tokens': len(ids), 'output_budget': budget}




class VisionPEFT:
    """Apply ALL language/vision adapter weights. Do not silently drop visual LoRA."""
    def __init__(self, c, pipeline):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import PeftModel
        processor_dir = c['ADAPTER_DIR'] if (Path(c['ADAPTER_DIR']) / 'preprocessor_config.json').exists() else c['MODEL_DIR']
        self.processor = AutoProcessor.from_pretrained(str(processor_dir))
        self.processor.image_processor.size = {'shortest_edge': c['MIN_IMAGE_PIXELS'],
                                                'longest_edge': c['MAX_IMAGE_PIXELS']}
        base = Qwen3VLForConditionalGeneration.from_pretrained(str(c['MODEL_DIR']),
            torch_dtype='auto' if c['DTYPE'] == 'auto' else getattr(torch, c['DTYPE']),
            device_map='auto', attn_implementation='sdpa')
        self.model = PeftModel.from_pretrained(base, str(c['ADAPTER_DIR']),
                                              is_trainable=False).eval()
        self.c, self.pipeline = c, pipeline
        torch.manual_seed(c['SEED'])

    def generate(self, messages, n):
        import copy
        import torch
        from PIL import Image
        c = self.c
        material = copy.deepcopy(messages)
        pictures = []
        for message in material:
            for block in message['content']:
                if block['type'] == 'image':
                    with Image.open(block['image']) as source:
                        picture = source.convert('RGB')
                    pictures.append(picture)
                    block['image'] = picture
        text = self.processor.apply_chat_template(material, tokenize=False, add_generation_prompt=True)
        kwargs = {'text': [text], 'return_tensors': 'pt', 'padding': True}
        if pictures:
            kwargs['images'] = pictures
        inputs = self.processor(**kwargs).to(self.model.get_input_embeddings().weight.device)
        input_tokens = inputs['input_ids'].shape[-1]
        budget = min(c['MAX_NEW_TOKENS'], c['MAX_MODEL_LEN'] - input_tokens)
        if budget <= 0:
            raise ValueError('Image + text tokens exceed MAX_MODEL_LEN')
        # Fifteen independent plans, in small microbatches to bound visual/KV memory.
        outputs = []
        tokenizer = self.processor.tokenizer
        eos = self.model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for start in range(0, n, c['VL_CANDIDATE_BATCH_SIZE']):
            with torch.inference_mode():
                result = self.model.generate(**inputs, do_sample=True,
                    temperature=c['TEMPERATURE'], top_p=c['TOP_P'], top_k=c['TOP_K'],
                    max_new_tokens=budget, num_return_sequences=min(c['VL_CANDIDATE_BATCH_SIZE'], n-start),
                    use_cache=True, repetition_penalty=1.0)
            for row in result:
                ids = row[input_tokens:].tolist()
                end = next((i + 1 for i, token in enumerate(ids) if token in eos), None)
                if end is not None:
                    ids = ids[:end]
                outputs.append(decode_candidate(tokenizer, ids, self.pipeline, False,
                                                'stop' if end is not None else 'length', input_tokens, budget))
            del result
        return outputs


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_trace(path):
    """A power loss can leave a partial last line; preserve and truncate that tail."""
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    done, consumed = {}, 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise ValueError('Corrupt trace before the final line')
            path.with_suffix('.interrupted_tail').write_bytes(line)
            path.write_bytes(data[:consumed])
            break
        done[row['qid']] = row
        consumed += len(line)
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as f:
            f.write(b'\n')
    return done


def make_repair_prompt(planner, official_image_labels, context, representatives):
    return planner + (
        "\nKhông có candidate hợp lệ. Đối chiếu bối cảnh và sinh lại toàn bộ kế hoạch."
        "\nLỗi table_at: dùng literal từ ô đã xác định đúng, rồi lập lại đủ phép tính;"
        " không thay bằng chart/tool khác để né lỗi. Các ảnh hợp lệ: " +
        json.dumps(official_image_labels, ensure_ascii=False) +
        "\nCác bảng hợp lệ: " + json.dumps(list(context["table"]), ensure_ascii=False) +
        "\nCác lỗi đại diện và đầu ra trước:\n" + json.dumps(representatives, ensure_ascii=False) +
        "\nĐầu ra trước là dữ liệu lỗi, không phải mẫu để sao chép. Chỉ xuất kế hoạch"
        " đã sửa trên các dòng thật, kết thúc bằng join() và <END_OF_PLAN>."
    )


def run(c, pipeline, planning, samples, out, backend_factory=None):
    adapter, hashes = adapter_info(c['ADAPTER_DIR'])
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in c.items()}
    contract.update(adapter_config=adapter, adapter_sha256=hashes,
                    prepared_inputs_hash=digest(samples))
    fingerprint = digest(contract)
    manifest = out / 'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['fingerprint'] != fingerprint:
        raise ValueError('Settings/adapter/input changed; use another OUTPUT_ROOT')
    atomic_json(manifest, {'fingerprint': fingerprint, 'config': contract})
    trace_path = out / 'trace.jsonl'
    done = read_trace(trace_path)
    qids = [s['qid'] for s in samples]
    if set(done) - set(qids):
        raise ValueError('Trace contains QIDs outside this shard')

    def export():
        records = [done[q] for q in qids if q in done]
        atomic_json(out / 'trace.json', records)
        atomic_json(out / 'predictions.json', [{'qid': r['qid'], 'predicted': r['predicted'],
                                               'status': r['status']} for r in records])
        atomic_json(out / 'results.json', [{'qid': r['qid'], 'program': r['predicted']} for r in records])
        labeled = [r for r in records if r.get('gold')]
        metrics = {'expected': len(samples), 'completed': len(records),
                   'ok': sum(r['status'] == 'ok' for r in records),
                   'errors': sum(r['status'] != 'ok' for r in records),
                   'n': c['N_PLANS'], 'labeled': len(labeled),
                   'strict_PA': sum(r['strict_match'] for r in labeled) / len(labeled) if labeled else None}
        atomic_json(out / 'metrics.json', metrics)
        return metrics

    pending = [s for s in samples if s['qid'] not in done]
    print('Resume:', len(done), 'saved;', len(pending), 'pending')
    if not pending:
        return export()
    backend = backend_factory() if backend_factory else VisionPEFT(c, pipeline)
    with trace_path.open('a', encoding='utf-8') as trace:
        for sample in pending:
            saved = {**sample, 'flow': c['FLOW'], 'model': str(c['MODEL_DIR']),
                     'adapter': str(c['ADAPTER_DIR']), 'shard_id': c['SHARD_ID'],
                     'n': c['N_PLANS'], 'fingerprint': fingerprint,
                     'predicted': '', 'candidates': [], 'votes': 0, 'status': 'error'}
            try:
                outputs = backend.generate(sample['input'], c['N_PLANS'])
                if len(outputs) != c['N_PLANS']:
                    raise RuntimeError('Backend returned an unexpected number of candidates')
                predicted, candidates, votes = planning.choose(outputs, sample['context'])
                for candidate in candidates:
                    candidate['phase'] = 'initial'
                saved['candidates'] = candidates
                # Preserve the original full pipeline's three-plan repair only when all 15 fail.
                if not predicted:
                    import copy
                    representatives, seen = [], set()
                    for candidate in candidates:
                        error = candidate.get('error', 'Invalid plan')
                        if error not in seen:
                            seen.add(error)
                            representatives.append({'error': error, 'raw': candidate.get('text', '')})
                        if len(representatives) == 3:
                            break
                    user = sample['input'][1]['content']
                    text = user if isinstance(user, str) else ''.join(b['text'] for b in user if b['type'] == 'text')
                    repair = make_repair_prompt(text, list(sample['context']['image_labels'].values()),
                                                sample['context'], representatives)
                    repair_input = copy.deepcopy(sample['input'])
                    if isinstance(user, str):
                        repair_input[1]['content'] = repair
                    else:
                        repair_input[1]['content'].append({'type': 'text', 'text': repair[len(text):]})
                    saved['planning_repair_input'] = repair_input
                    repaired = backend.generate(repair_input, 3)
                    predicted, repaired_candidates, votes = planning.choose(repaired, sample['context'])
                    for candidate in repaired_candidates:
                        candidate.update(index=len(candidates), phase='repair')
                        candidates.append(candidate)
                saved.update(predicted=predicted, candidates=candidates, votes=votes,
                             valid_candidates=sum('steps' in c for c in candidates),
                             status='ok' if predicted else 'error')
            except Exception as exc:
                saved.update(predicted='', votes=0, status='error',
                             error=type(exc).__name__ + ': ' + str(exc))
            saved['strict_match'] = planning.strict_match(saved['predicted'], sample['gold']) if sample.get('gold') else None
            trace.write(json.dumps(saved, ensure_ascii=False) + '\n')
            trace.flush()
            os.fsync(trace.fileno())
            done[sample['qid']] = saved
            print(export(), flush=True)
            print('QID:', sample['qid'], '| votes:', saved['votes'], '/', c['N_PLANS'])
            print('PREDICTED:', saved['predicted'], flush=True)
    return export()