import json
import sys
import zipfile
from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent
CODE_DIR = FINAL_DIR / 'common'
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

if __name__ == "__main__":
    import torch
    from inference_engine import prepare, run, adapter_info
    SPLIT = 'private'
    if SPLIT not in ('public', 'private'):
        raise ValueError('SPLIT must be public or private')

    FLOW = 'vl_full'
    MODEL_DIR = 'Qwen/Qwen3-VL-8B-Instruct'
    BGE_MODEL_DIR = 'BAAI/bge-m3'
    ADAPTER_DIR = FINAL_DIR / 'models' / 'lora'
    OUTPUT_ROOT = FINAL_DIR / 'outputs' / f'qwen3_vl_infer_{SPLIT}'

    INPUT_FILE = FINAL_DIR / 'data' / f'{SPLIT}_test.json'
    EVIDENCE_FILE = FINAL_DIR / 'data' / 'decompose_extraction_evidence' / f'dec_ex_{SPLIT}.jsonl'
    CHART_CAPTION_FILE = FINAL_DIR / 'data' / 'caption' / f'{SPLIT}_test_caption.jsonl'
    IMAGE_ROOT = FINAL_DIR / 'data' / f'{SPLIT}_test_images'
    TRAIN_FILE = FINAL_DIR / 'data' / 'train.json'

    if not (ADAPTER_DIR / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Place your trained adapter in {ADAPTER_DIR}; it is not a public base model')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for VL inference')

    required = [CODE_DIR / 'inference_engine.py', ADAPTER_DIR / 'adapter_config.json',
                INPUT_FILE, CHART_CAPTION_FILE, TRAIN_FILE]
    required.extend([EVIDENCE_FILE, IMAGE_ROOT])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required input files:\n' + '\n'.join(missing))

    adapter_config, _ = adapter_info(ADAPTER_DIR)
    print('Backend: Transformers + PEFT')
    print(f'FLOW: {FLOW} | BASE: {MODEL_DIR} | ADAPTER: {ADAPTER_DIR}')
    print('Adapter training base:', adapter_config.get('base_model_name_or_path'))

    CONFIG = {
        'FLOW': FLOW, 'CODE_DIR': CODE_DIR, 'MODEL_DIR': MODEL_DIR, 'ADAPTER_DIR': ADAPTER_DIR,
        'INPUT_FILE': INPUT_FILE, 'EVIDENCE_FILE': EVIDENCE_FILE, 'CHART_CAPTION_FILE': CHART_CAPTION_FILE,
        'IMAGE_ROOT': IMAGE_ROOT, 'TRAIN_FILE': TRAIN_FILE, 'BGE_MODEL_DIR': BGE_MODEL_DIR,
        'OUTPUT_ROOT': OUTPUT_ROOT, 'N_SHARDS': 1, 'SHARD_ID': 0, 'LIMIT': None, 'N_PLANS': 15,
        'RETRIEVAL_TOP_K': 5, 'MAX_NEW_TOKENS': 16384, 'DTYPE': 'bfloat16', 'SEED': 0,
        'TEMPERATURE': 0.7, 'TOP_P': 0.8, 'TOP_K': 20, 'ENABLE_THINKING': False,
    }
    CONFIG.update(MAX_MODEL_LEN=40960, MIN_IMAGE_PIXELS=256 * 256, MAX_IMAGE_PIXELS=1024 * 1024,
                  VL_CANDIDATE_BATCH_SIZE=3)

    pipeline, planning, samples, OUTPUT_DIR = prepare(CONFIG)
    print('Output directory:', OUTPUT_DIR)
    metrics = run(CONFIG, pipeline, planning, samples, OUTPUT_DIR)
    print(json.dumps(metrics, indent=2))

    zip_path = OUTPUT_DIR.parent / (OUTPUT_DIR.name + '.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('results.json', 'predictions.json', 'trace.json', 'trace.jsonl',
                     'metrics.json', 'run_manifest.json', 'prepared_inputs.jsonl'):
            path = OUTPUT_DIR / name
            if path.exists():
                archive.write(path, arcname=OUTPUT_DIR.name + '/' + name)
    print('Download archive:', zip_path)