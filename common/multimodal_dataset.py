import copy
import hashlib
import io
import json
import random
import re
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

from program_format import submission_to_planner
from pipeline import normalize_sample

VARIANTS = ('base', 'swapped', 'decompose', 'no_subquery')
IMAGE_MARKER = re.compile(r'###\s*(Image\s+\d+)\s*###', re.I)
TEXT_ONLY = 'Model chỉ nhận text, bảng và caption, không xem ảnh gốc.'
CAPTION_ONLY = 'Caption dưới marker thuộc ảnh đó; có thể thiếu/sai, không đoán series, nhãn hay số.'
VISION_INPUT = 'Model nhận ảnh gốc, text, bảng và caption.'
VISION_GUIDE = (
    'Ảnh ngay sau marker thuộc ID đó; không đánh lại số ảnh theo thứ tự xuất hiện.\n'
    'Caption/evidence chỉ là gợi ý; nếu mâu thuẫn, đối chiếu ảnh gốc và nguồn hiện tại. '
    'Ưu tiên thông tin đọc rõ từ ảnh, không đoán series, nhãn hay giá trị mờ.'
)

def build_context(raw_row, captions):
    row = normalize_sample(raw_row)
    tables = row.get("table") or {}
    if isinstance(tables, str):
        tables = {"table_0": tables}
    elif isinstance(tables, list):
        tables = {f"table_{i}": v for i, v in enumerate(tables)}
    images = row.get("image") or {}
    if isinstance(images, str):
        images = [images]
    if isinstance(images, list):
        images = {f"image_{i}": v for i, v in enumerate(images)}
    labels = {f"image_{i}": str(k) for i, k in enumerate(images)}
    chart = {}
    for label, filename in images.items():
        name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
        if captions.get(name):
            chart[str(label)] = {"image_file": name, "raw_output": captions[name]}
    context = {"question": row["question"], "text": row.get("text", []),
               "table": tables, "image_ids": list(labels), "image_labels": labels,
               "executable_html_table_ids": [k for k, v in tables.items()
                   if isinstance(v, str) and re.search(r"<(table|tr)\b", v, re.I)]}
    if chart:
        context["chart_perception"] = chart
    return context

def patch_system(system):
    if not isinstance(system, str) or not system.strip():
        raise ValueError('Missing system prompt')
    # Exact replacements only: other rules and differences between variants survive.
    if TEXT_ONLY not in system or CAPTION_ONLY not in system:
        raise ValueError('Unexpected source system prompt; inspect before applying VL changes')
    return system.replace(TEXT_ONLY, VISION_INPUT).replace(CAPTION_ONLY, VISION_GUIDE)


def load_raw_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if Path(n).name == 'train.json']
            if len(names) != 1:
                raise ValueError('TRAIN_FILE zip must contain exactly one train.json')
            data = json.loads(archive.read(names[0]).decode('utf-8-sig'))
    else:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = data['root'] if isinstance(data, dict) and isinstance(data.get('root'), list) else data
    if not isinstance(rows, list):
        raise ValueError('Expected train list or {root: list}')
    by_id = {}
    for row in rows:
        qid = str(row.get('qid') or row.get('id') or '')
        if not qid or qid in by_id:
            raise ValueError('Missing or duplicated train QID: ' + qid)
        by_id[qid] = row
    return by_id


def iter_variants(augment_path):
    """Accept the supplied ZIP or its extracted vlsp-data directory."""
    augment_path = Path(augment_path)
    if augment_path.is_file() and augment_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(augment_path) as archive:
            for variant in VARIANTS:
                names = [n for n in archive.namelist() if Path(n).name == variant + '.jsonl']
                if len(names) != 1:
                    raise ValueError(f'Expected exactly one {variant}.jsonl in ZIP')
                with archive.open(names[0]) as binary:
                    for line_no, line in enumerate(io.TextIOWrapper(binary, encoding='utf-8-sig'), 1):
                        if line.strip():
                            yield variant, line_no, json.loads(line)
    else:
        for variant in VARIANTS:
            candidates = [augment_path / (variant + '.jsonl'), augment_path / 'vlsp-data' / (variant + '.jsonl')]
            found = [p for p in candidates if p.is_file()]
            if len(found) != 1:
                raise FileNotFoundError(f'Locate {variant}.jsonl under {augment_path}')
            with found[0].open(encoding='utf-8-sig') as stream:
                for line_no, line in enumerate(stream, 1):
                    if line.strip():
                        yield variant, line_no, json.loads(line)


def image_map(raw):
    images = raw.get('images', raw.get('image', {})) or {}
    if isinstance(images, list):
        images = {f'Image {i}': value for i, value in enumerate(images, 1)}
    if not isinstance(images, dict):
        raise ValueError('Unexpected original image mapping')
    result = {}
    for label, filename in images.items():
        match = re.fullmatch(r'Image\s+(\d+)', str(label), re.I)
        if not match or not isinstance(filename, str):
            raise ValueError(f'Image label/path must come from the raw source: {label}')
        canonical = f'Image {int(match[1])}'
        if canonical in result:
            raise ValueError('Duplicate image label: ' + canonical)
        result[canonical] = filename
    return result


def index_images(image_root):
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f'IMAGE_ROOT is not a directory: {image_root}')
    index = {}
    for path in image_root.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
            if path.name in index and path.resolve() != index[path.name].resolve():
                raise ValueError('Ambiguous original image filename: ' + path.name)
            index[path.name] = path
    return index


def multimodal_user(text, mapping, image_index, *, allow_unmapped_markers=False):
    """Keep original text byte-for-byte and attach each ID's pixels at its marker."""
    if not isinstance(text, str):
        raise ValueError('Expected a text user prompt in the augmentation')
    content, seen, cursor = [], set(), 0
    for match in IMAGE_MARKER.finditer(text):
        label = 'Image ' + str(int(re.search(r'\d+', match[1])[0]))
        if label not in mapping:
            if allow_unmapped_markers:
                # Preserve unmatched raw markers as text; never invent pixels or IDs.
                continue
            raise ValueError(f'Marker {label} has no corresponding image in original train row')
        content.append({'type': 'text', 'text': text[cursor:match.end()]})
        if label not in seen:
            filename = Path(mapping[label].replace('\\', '/')).name
            if filename not in image_index:
                raise FileNotFoundError('Missing original image: ' + filename)
            content.append({'type': 'image', 'image': str(image_index[filename].resolve())})
            seen.add(label)
        cursor = match.end()
    if cursor < len(text) or not content:
        content.append({'type': 'text', 'text': text[cursor:]})
    missing = set(mapping) - seen
    if missing:
        # Do not silently add unlabeled pictures or rebuild the context.
        raise ValueError('Original image has no marker in user context: ' + ', '.join(sorted(missing)))
    if ''.join(c['text'] for c in content if c['type'] == 'text') != text:
        raise RuntimeError('User context changed during image insertion')
    return content


def make_input_messages(messages, raw, image_index, *, allow_unmapped_markers=False):
    """Identical system/user transformation for SFT and inference; no gold access."""
    if not isinstance(messages, list) or [m.get('role') for m in messages] != ['system', 'user']:
        raise ValueError('Expected exactly system/user input messages')
    content = multimodal_user(messages[1]['content'], image_map(raw), image_index,
                              allow_unmapped_markers=allow_unmapped_markers)
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': patch_system(messages[0]['content'])}]},
        {'role': 'user', 'content': content},
    ]


def make_record(saved, raw, variant, image_index):
    gold = (raw.get('qa') or {}).get('program') or raw.get('program', '')
    target, error = submission_to_planner(gold)
    if target is None:
        raise ValueError('Cannot format original gold: ' + str(error))
    return {
        'qid': str(raw.get('qid') or raw.get('id')), 'variant': variant,
        'messages': make_input_messages(saved.get('input'), raw, image_index) + [
            {'role': 'assistant', 'content': [{'type': 'text', 'text': target}]},
        ],
    }


def prepare_records(augment_path, train_file, image_root):
    raw = load_raw_rows(train_file)
    images = index_images(image_root)
    seen = {v: set() for v in VARIANTS}
    records, errors = [], []
    provenance = Counter()
    systems = set()
    for variant, line_no, saved in iter_variants(augment_path):
        qid = str(saved.get('qid') or saved.get('id') or '')
        provenance[str(saved.get('model'))] += 1
        if not qid or qid in seen[variant]:
            raise ValueError(f'Duplicate/missing QID in {variant}: {qid}')
        seen[variant].add(qid)
        try:
            if qid not in raw:
                raise ValueError('QID not in original train')
            record = make_record(saved, raw[qid], variant, images)
            records.append(record)
            systems.add(record['messages'][0]['content'][0]['text'])
        except (ValueError, FileNotFoundError) as exc:
            errors.append({'qid': qid, 'variant': variant, 'line': line_no, 'error': str(exc)})
    coverage = {v: {'count': len(ids), 'missing_train_qids': sorted(set(raw) - ids),
                    'unknown_qids': sorted(ids - set(raw))} for v, ids in seen.items()}
    manifest = {'variant_coverage': coverage, 'prepared_rows': len(records),
                'unique_qids': len({r['qid'] for r in records}), 'errors': len(errors),
                'image_count_histogram': dict(Counter(sum(c['type'] == 'image' for c in r['messages'][1]['content']) for r in records)),
                'input_record_model_metadata': dict(provenance),
                'system_prompts': sorted(systems), 'target': 'original train gold program only; no reasoning',
                'user_change': 'image content at existing markers; original text preserved'}
    return records, errors, manifest


def split_by_qid(records, validation_qids=0, seed=42):
    ids = sorted({r['qid'] for r in records})
    if not 0 <= validation_qids < len(ids):
        raise ValueError('validation_qids must leave training QIDs')
    random.Random(seed).shuffle(ids)
    valid = set(ids[:validation_qids])
    train_rows = [r for r in records if r['qid'] not in valid]
    eval_rows = [r for r in records if r['qid'] in valid]
    random.Random(seed).shuffle(train_rows)
    return train_rows, eval_rows


@lru_cache(maxsize=32)
def load_rgb(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.convert('RGB')


def materialize(record):
    item = copy.deepcopy(record)
    for message in item['messages']:
        for content in message['content']:
            if content['type'] == 'image':
                content['image'] = load_rgb(content['image']).copy()
    return item


def records_digest(records):
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()