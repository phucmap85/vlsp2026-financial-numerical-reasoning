import inspect
import json
import re
import sys
from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent
CODE_DIR = FINAL_DIR / 'common'
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    class Dataset:
        pass
from multimodal_dataset import prepare_records, split_by_qid, records_digest, materialize, VARIANTS


class GoldVisionDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return materialize(self.rows[index])


class CheckedGoldCollator:
    def __init__(self, collator, tokenizer, max_length):
        self.collator = collator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_ids = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)

    def __call__(self, features):
        expected = [r['messages'][-1]['content'][0]['text'] for r in features]
        batch = self.collator(features)
        expected_images = sum(c['type'] == 'image' for r in features for m in r['messages'] for c in m['content'])
        if expected_images:
            grid = batch.get('image_grid_thw')
            if batch.get('pixel_values') is None or grid is None or grid.shape[0] != expected_images:
                raise ValueError('Image tensors do not match the original Image N content blocks')
        if batch['input_ids'].shape[1] > self.max_length:
            raise ValueError(f'Full multimodal length {batch["input_ids"].shape[1]} exceeds {self.max_length} '
                             f'for {[r["qid"] for r in features]}; increase budget or lower image resolution.')

        for i, target in enumerate(expected):
            ids = batch['input_ids'][i].tolist()
            labels = batch['labels'][i]
            marker = self.response_ids
            positions = [j for j in range(len(ids) - len(marker) + 1) if ids[j:j+len(marker)] == marker]
            if not positions:
                raise ValueError('Assistant header not found in the actual processor tokens')
            start = positions[-1] + len(marker)
            if (labels[:start] != -100).any().item():
                raise ValueError('System/user/image tokens must be masked from the loss')
            trainable = labels[labels != -100].tolist()
            if not trainable:
                raise ValueError('No assistant gold tokens to train')
            decoded = self.tokenizer.decode(trainable, skip_special_tokens=True)
            if re.sub(r'\s+', '', decoded) != re.sub(r'\s+', '', target):
                raise ValueError('Supervised tokens do not equal the complete gold program')
        return batch


if __name__ == "__main__":
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers.trainer_utils import get_last_checkpoint

    AUGMENT_PATH = FINAL_DIR / 'data' / 'augment'
    TRAIN_FILE = FINAL_DIR / 'data' / 'train.json'
    IMAGE_ROOT = FINAL_DIR / 'data' / 'train_images'
    MODEL_DIR = 'Qwen/Qwen3-VL-8B-Instruct'
    OUTPUT_DIR = FINAL_DIR / 'outputs' / 'qwen3_vl_sft'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    SEED = 42
    VALIDATION_QIDS = 0
    USE_QLORA = False
    MAX_SEQ_LENGTH = 32768
    MIN_IMAGE_PIXELS = 256 * 256
    MAX_IMAGE_PIXELS = 1024 * 1024
    LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
    FINETUNE_VISION_LAYERS = True
    NUM_EPOCHS = 3
    PER_DEVICE_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16
    LEARNING_RATE = 2e-5
    SAVE_STEPS = 50
    SAVE_TOTAL_LIMIT = None
    RESUME_FROM_CHECKPOINT = 'auto'

    records, errors, data_manifest = prepare_records(AUGMENT_PATH, TRAIN_FILE, IMAGE_ROOT)
    (OUTPUT_DIR / 'data_manifest.json').write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'data_errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
    if errors:
        raise ValueError(f'{len(errors)} image/prompt/gold joins failed; see data_errors.json')
    if any(v['missing_train_qids'] or v['unknown_qids'] for v in data_manifest['variant_coverage'].values()):
        raise ValueError('Each variant must cover the original train QIDs; see data_manifest.json')

    train_records, eval_records = split_by_qid(records, VALIDATION_QIDS, SEED)
    print('Variants:', data_manifest['variant_coverage'])
    print(f'Rows: {len(records)} | Unique QIDs: {data_manifest["unique_qids"]}')
    print(f'Train rows: {len(train_records)} | Eval rows: {len(eval_records)}')

    with (OUTPUT_DIR / 'prepared_gold_multimodal.jsonl').open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    manifest = {
        'model': str(MODEL_DIR), 'target': 'gold program only', 'thinking': False,
        'variants': list(VARIANTS), 'records_sha256': records_digest(records),
        'train_qids': sorted({r['qid'] for r in train_records}),
        'eval_qids': sorted({r['qid'] for r in eval_records}),
        'epochs': NUM_EPOCHS, 'seed': SEED, 'max_seq_length': MAX_SEQ_LENGTH,
        'image_pixels': [MIN_IMAGE_PIXELS, MAX_IMAGE_PIXELS], 'use_qlora': USE_QLORA,
        'lora': [LORA_R, LORA_ALPHA, LORA_DROPOUT, FINETUNE_VISION_LAYERS],
        'batch_size': PER_DEVICE_BATCH_SIZE, 'gradient_accumulation': GRADIENT_ACCUMULATION_STEPS,
        'learning_rate': LEARNING_RATE, 'source_data': str(AUGMENT_PATH),
    }
    manifest_path = OUTPUT_DIR / 'experiment_manifest.json'
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Data/training settings changed; use a fresh OUTPUT_DIR instead of resuming old checkpoints.')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for training.')

    model, processor = FastVisionModel.from_pretrained(
        model_name=str(MODEL_DIR), load_in_4bit=USE_QLORA, dtype=torch.bfloat16,
        use_gradient_checkpointing='unsloth', max_seq_length=MAX_SEQ_LENGTH,
        trust_remote_code=False, device_map={'': 0})

    image_processor = processor.image_processor
    if isinstance(getattr(image_processor, 'size', None), dict):
        image_processor.size = dict(image_processor.size, shortest_edge=MIN_IMAGE_PIXELS, longest_edge=MAX_IMAGE_PIXELS)
    image_processor.min_pixels = MIN_IMAGE_PIXELS
    image_processor.max_pixels = MAX_IMAGE_PIXELS

    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastVisionModel.get_peft_model(model,
        finetune_vision_layers=FINETUNE_VISION_LAYERS, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias='none', random_state=SEED, use_rslora=False, loftq_config=None)
    FastVisionModel.for_training(model)
    model.print_trainable_parameters()

    collator_kwargs = dict(train_on_responses_only=True,
        instruction_part='<|im_start|>user\n', response_part='<|im_start|>assistant\n',
        completion_only_loss=True, resize='max')
    if 'last_response_only' in inspect.signature(UnslothVisionDataCollator).parameters:
        collator_kwargs['last_response_only'] = True
    vision_collator = UnslothVisionDataCollator(model, processor, **collator_kwargs)
    vision_collator.max_seq_length = None
    vision_collator.truncation = False

    collator = CheckedGoldCollator(vision_collator, tokenizer, MAX_SEQ_LENGTH)
    train_dataset = GoldVisionDataset(train_records)
    eval_dataset = GoldVisionDataset(eval_records) if eval_records else None

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR), num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE, per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, learning_rate=LEARNING_RATE,
        warmup_ratio=0.05, weight_decay=0.01, lr_scheduler_type='cosine', optim='adamw_8bit',
        bf16=True, fp16=False, logging_steps=10, save_strategy='steps', save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT, save_only_model=False,
        eval_strategy='steps' if eval_dataset is not None else 'no',
        eval_steps=SAVE_STEPS if eval_dataset is not None else None,
        report_to='none', seed=SEED, data_seed=SEED, gradient_checkpointing=True,
        remove_unused_columns=False, packing=False, dataset_text_field='',
        dataset_kwargs={'skip_prepare_dataset': True}, max_length=None, dataloader_num_workers=0)

    trainer_kwargs = dict(model=model, args=training_args, train_dataset=train_dataset,
                          eval_dataset=eval_dataset, data_collator=collator)
    if 'processing_class' in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs['processing_class'] = processor
    else:
        trainer_kwargs['tokenizer'] = processor
    trainer = SFTTrainer(**trainer_kwargs)

    resume = get_last_checkpoint(str(OUTPUT_DIR)) if RESUME_FROM_CHECKPOINT == 'auto' else RESUME_FROM_CHECKPOINT
    print('Resume checkpoint:', resume)
    train_result = trainer.train(resume_from_checkpoint=resume)

    adapter_out_dir = FINAL_DIR / 'models' / 'lora'
    adapter_out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_out_dir))
    processor.save_pretrained(str(adapter_out_dir))
    trainer.save_state()
    (adapter_out_dir / 'train_metrics.json').write_text(json.dumps(train_result.metrics, indent=2, default=str), encoding='utf-8')
    (adapter_out_dir / 'vision_prompt_contract.json').write_text(json.dumps({
        'system_prompts': data_manifest['system_prompts'], 'thinking': False,
        'image_order': 'Original Image N markers; never renumber by appearance',
        'target': 'gold program only', 'inference_flow': 'decomposition + extraction + planning with retrieval/images/captions',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    history = trainer.state.log_history
    (OUTPUT_DIR / 'loss_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    train_logs = [r for r in history if 'loss' in r]
    eval_logs = [r for r in history if 'eval_loss' in r]
    print('Saved adapter and processor:', adapter_out_dir)