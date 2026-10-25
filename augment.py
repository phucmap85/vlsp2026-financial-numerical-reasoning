import json
import random
import re
import sys
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_unique(rows):
    result = {}
    for row in rows:
        qid = str(row.get("qid") or row.get("id") or "")
        if not qid or qid in result:
            raise ValueError("Missing/duplicate QID: " + qid)
        result[qid] = row
    return result


def build_base(final_dir):
    """Build base using hybrid retrieval and planner few-shots."""
    code_dir = final_dir / "common"
    train_file = final_dir / "data" / "train.json"
    caption_file = final_dir / "data" / "caption" / "train_caption.jsonl"
    evidence_file = final_dir / "data" / "decompose_extraction_evidence" / "dec_ex_train.jsonl"
    bge_model = "BAAI/bge-m3"
    cache_dir = final_dir / "outputs" / "build_base_cache"
    output_file = final_dir / "data" / "augment" / "base.jsonl"

    for path in (train_file, caption_file, evidence_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import pipeline
    from prompt_contract import planner_user, render_inline_context, select_examples
    from multimodal_dataset import build_context
    from hybrid_retriever import HybridRetriever

    raw = json.loads(train_file.read_text(encoding="utf-8-sig"))
    train = raw if isinstance(raw, list) else raw["root"]
    evidence = index_unique(read_jsonl(evidence_file))
    train_ids = {str(pipeline.normalize_sample(r)["id"]) for r in train}
    assert len(train_ids) == len(train)
    assert train_ids == set(evidence), "QID coverage mismatch"
    captions = pipeline.load_chart_captions(caption_file)

    targets = {}
    for row in train:
        norm = pipeline.normalize_sample(row)
        qid = str(norm["id"])
        gold = (row.get("qa") or {}).get("program") or row.get("program")
        target, error = pipeline.submission_to_planner(gold)
        if target is None:
            raise ValueError(f"Cannot format gold for {qid}: {error}")
        targets[qid] = target

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_corpus_path = cache_dir / "train_corpus.json"
    corpus = [{"id": str(pipeline.normalize_sample(r)["id"]), "qa": {
        "question": pipeline.normalize_sample(r)["question"],
        "program": (r.get("qa") or {}).get("program") or r.get("program", "")
    }} for r in train]
    train_corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    index_by_id = {row["id"]: i for i, row in enumerate(corpus)}
    embeddings_path = cache_dir / "bge_m3_train_embeddings.pt"
    retriever = HybridRetriever(str(train_corpus_path), bge_model, str(embeddings_path))

    records = []
    for index, row in enumerate(train):
        norm = pipeline.normalize_sample(row)
        qid = str(norm["id"])
        hits = retriever.search(norm["question"], top_k=10, exclude_index=index_by_id[qid])
        hits = [dict(h, qid=corpus[h["index"]]["id"], program=h["answer"]) for h in hits]
        examples, audit = select_examples(hits, norm["question"], qid, pipeline.submission_to_planner, top_k=5)
        ev = evidence[qid]
        extracted = ev.get("extractions", ev.get("evidence"))
        queries = ev.get("subqueries")
        if not isinstance(extracted, list) or not isinstance(queries, list) or len(extracted) != len(queries):
            raise ValueError("Invalid evidence: " + qid)
        if any(e.get("query") != q or not isinstance(e.get("answer"), str) or not e["answer"].strip()
               for q, e in zip(queries, extracted)):
            raise ValueError("Evidence query/answer mismatch: " + qid)
        context = build_context(row, captions)
        missing = set(context["image_labels"].values()) - set(context.get("chart_perception", {}))
        if missing:
            raise ValueError(("Missing captions", qid, missing))
        user = planner_user(render_inline_context(context), norm["question"], extracted, examples)
        user = re.sub(r"<think>.*?</think>", "", user, flags=re.DOTALL).strip()
        predicted = targets.get(qid, "")
        predicted = re.sub(r"<think>.*?</think>", "", predicted, flags=re.DOTALL).strip()
        messages = [{"role": "system", "content": pipeline.PLANNING_SYSTEM}, {"role": "user", "content": user}]
        record = {"qid": qid, "input": messages, "predicted": predicted}
        serialized = json.dumps(record, ensure_ascii=False)
        if "<think>" in serialized or "</think>" in serialized:
            raise ValueError("Unexpected think tag: " + qid)
        records.append(record)
        if (index + 1) % 100 == 0 or index + 1 == len(train):
            print(f"Prepared base: {index + 1}/{len(train)}", flush=True)

    temporary = output_file.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_file)
    print("Saved:", output_file)


def extract_ctx_blocks(ctx_text):
    """Segment context text into media blocks (table/image) and pure text blocks."""
    markers = list(re.finditer(r"(### (?:Table|Image) \d+ ###)", ctx_text))
    if not markers:
        return [], [ctx_text]
    blocks = []
    first_text = ctx_text[:markers[0].start()].strip()
    if first_text:
        blocks.append({"is_media": False, "content": first_text})
    for i, m in enumerate(markers):
        hdr = m.group(1)
        start = m.start()
        next_m_start = markers[i+1].start() if i+1 < len(markers) else len(ctx_text)
        chunk = ctx_text[start:next_m_start].strip()
        if "Table" in hdr:
            tbl_end = chunk.find("</table>")
            if tbl_end != -1:
                media_content = chunk[:tbl_end + len("</table>")].strip()
                trailing_text = chunk[tbl_end + len("</table>"):].strip()
            else:
                media_content = chunk
                trailing_text = ""
        else:
            dl_match = re.search(r"(\nDATA_LABELS:[^\n]+(?:\n\nNguồn:[^\n]+)?)", chunk)
            if dl_match:
                end_idx = dl_match.end()
                media_content = chunk[:end_idx].strip()
                trailing_text = chunk[end_idx:].strip()
            else:
                media_content = chunk
                trailing_text = ""
        blocks.append({"is_media": True, "content": media_content})
        if trailing_text:
            blocks.append({"is_media": False, "content": trailing_text})
    media_blocks = [b["content"] for b in blocks if b["is_media"]]
    text_blocks = [b["content"] for b in blocks if not b["is_media"]]
    return media_blocks, text_blocks


def transform_swapped_sample(item, seed):
    rng = random.Random(seed)
    qid = item["qid"]
    sys_prompt = next(m["content"] for m in item["input"] if m["role"] == "system")
    u_text = next(m["content"] for m in item["input"] if m["role"] == "user")
    predicted = item.get("predicted") or item.get("output", "")
    predicted = re.sub(r"<think>.*?</think>", "", predicted, flags=re.DOTALL).strip()

    m1 = re.search(r"BỐI CẢNH HIỆN TẠI:\s*", u_text)
    m2 = re.search(r"\n\s*CÂU HỎI HIỆN TẠI:\s*", u_text)
    m3 = re.search(r"\n\s*BỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON:\s*", u_text)
    m4 = re.search(r"\n\s*VÍ DỤ MINH HỌA:\s*", u_text)
    assert m1 and m2 and m3 and m4, f"Missing section in qid: {qid}"

    ctx_text = u_text[m1.end():m2.start()].strip()
    q_text = u_text[m2.end():m3.start()].strip()
    subq_text = u_text[m3.end():m4.start()].strip()
    fewshot_text = u_text[m4.end():].strip()

    media_blocks, text_blocks = extract_ctx_blocks(ctx_text)
    rng.shuffle(media_blocks)
    layout_mode = rng.choice(["inplace", "top", "bottom"])
    if layout_mode == "top":
        if len(text_blocks) > 0 and len(text_blocks[0]) < 200:
            new_ctx = "\n\n".join([text_blocks[0]] + media_blocks + text_blocks[1:])
        else:
            new_ctx = "\n\n".join(media_blocks + text_blocks)
    elif layout_mode == "bottom":
        new_ctx = "\n\n".join(text_blocks + media_blocks)
    else:
        new_ctx = "\n\n".join(media_blocks + text_blocks)

    subq_parts = re.split(r"(?:^|\n)\s*(Truy vấn \d+:)", subq_text)
    subqs = [subq_parts[i+1].strip() for i in range(1, len(subq_parts), 2)]
    rng.shuffle(subqs)
    new_subq_text = "\n\n".join(f"Truy vấn {idx}: {sq}" for idx, sq in enumerate(subqs, 1))

    few_parts = re.split(r"(?:^|\n)\s*(Ví dụ \d+:)", fewshot_text)
    fewshots = [few_parts[i+1].strip() for i in range(1, len(few_parts), 2)]
    rng.shuffle(fewshots)
    new_fewshot_text = "\n\n".join(f"Ví dụ {idx}:\n{fs}" for idx, fs in enumerate(fewshots, 1))

    new_user_prompt = (
        "BỐI CẢNH HIỆN TẠI:\n"
        f"{new_ctx}\n\n"
        f"CÂU HỎI HIỆN TẠI: {q_text}\n\n"
        "BỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON:\n"
        f"{new_subq_text}\n\n"
        "VÍ DỤ MINH HỌA:\n"
        f"{new_fewshot_text}"
    )
    return {
        "qid": qid,
        "input": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": new_user_prompt}
        ],
        "predicted": predicted
    }


def gen_swapped(data_dir):
    """Generate swapped by permuting media blocks and subqueries."""
    data_dir = Path(data_dir)
    src_path = data_dir / "base.jsonl"
    out_path = data_dir / "swapped.jsonl"
    print(f"Generating {out_path.name} from {src_path.name}...")
    count = 0
    with open(src_path, "r", encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for idx, line in enumerate(in_f):
            if not line.strip():
                continue
            item = json.loads(line)
            transformed = transform_swapped_sample(item, seed=idx + 2026)
            out_f.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            count += 1
    print(f"Done swapped: wrote {count} rows to {out_path}")


def convert_to_direct_sys(base_sys):
    return base_sys.replace(
        "Đối chiếu câu hỏi gốc và evidence trung gian với context; evidence có thể sai.",
        "Xác định các đầu vào từ context gốc, giữ đúng đối tượng, kỳ và đơn vị; không suy đoán giá trị hoặc quan hệ mà nguồn không cung cấp."
    ).replace("caption/extraction", "caption")


def gen_no_subquery(data_dir):
    """Generate direct context without subqueries."""
    data_dir = Path(data_dir)
    src_path = data_dir / "swapped.jsonl" if (data_dir / "swapped.jsonl").exists() else data_dir / "base.jsonl"
    out_path = data_dir / "no_subquery.jsonl"
    subquery_pattern = re.compile(r"\n\nBỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON:.*?(?=\n\nVÍ DỤ MINH HỌA:)", re.DOTALL)
    print(f"Generating {out_path.name} from {src_path.name}...")
    count = 0
    with open(src_path, "r", encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            item = json.loads(line)
            base_sys = next(m["content"] for m in item["input"] if m["role"] == "system")
            user_prompt = next(m["content"] for m in item["input"] if m["role"] == "user")
            predicted = item.get("predicted") or item.get("output", "")
            predicted = re.sub(r"<think>.*?</think>", "", predicted, flags=re.DOTALL).strip()
            direct_sys = convert_to_direct_sys(base_sys)
            no_subquery_user_prompt = subquery_pattern.sub("", user_prompt)
            record = {
                "qid": item["qid"],
                "input": [
                    {"role": "system", "content": direct_sys},
                    {"role": "user", "content": no_subquery_user_prompt}
                ],
                "predicted": predicted
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Done no_subquery: wrote {count} rows to {out_path}")


def convert_to_decomp_sys(base_sys):
    return base_sys.replace(
        "Đối chiếu câu hỏi gốc và evidence trung gian với context; evidence có thể sai.",
        "Xác định các đầu vào từ context gốc, giữ đúng đối tượng, kỳ và đơn vị; không suy đoán giá trị hoặc quan hệ mà nguồn không cung cấp.\nCác truy vấn gợi ý có thể thiếu hoặc sai; ưu tiên câu hỏi gốc và context, tự bổ sung hoặc bỏ truy vấn khi cần, không coi chúng là dữ kiện."
    ).replace("caption/extraction", "caption")


def gen_decompose(data_dir):
    """Generate subqueries without intermediate extractions."""
    data_dir = Path(data_dir)
    src_path = data_dir / "swapped.jsonl" if (data_dir / "swapped.jsonl").exists() else data_dir / "base.jsonl"
    out_path = data_dir / "decompose.jsonl"
    print(f"Generating {out_path.name} from {src_path.name}...")
    count = 0
    with open(src_path, "r", encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            item = json.loads(line)
            base_sys = next(m["content"] for m in item["input"] if m["role"] == "system")
            u_text = next(m["content"] for m in item["input"] if m["role"] == "user")
            predicted = item.get("predicted") or item.get("output", "")
            predicted = re.sub(r"<think>.*?</think>", "", predicted, flags=re.DOTALL).strip()
            decomp_sys = convert_to_decomp_sys(base_sys)
            prefix, rest = u_text.split("\n\nBỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON:\n", 1)
            sub_raw, fs_raw = rest.split("\n\nVÍ DỤ MINH HỌA:\n", 1)
            queries = re.findall(r"^Truy vấn \d+:\s*(.*)$", sub_raw, re.MULTILINE)
            new_sub = "GỢI Ý CÁC THÔNG TIN CẦN TÌM:\n" + "\n".join(f"{i+1}. {q.strip()}" for i, q in enumerate(queries))
            decompose_user_prompt = prefix + "\n\n" + new_sub + "\n\nVÍ DỤ MINH HỌA:\n" + fs_raw
            record = {
                "qid": item["qid"],
                "input": [
                    {"role": "system", "content": decomp_sys},
                    {"role": "user", "content": decompose_user_prompt}
                ],
                "predicted": predicted
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Done decompose: wrote {count} rows to {out_path}")


def main():
    final_dir = Path(__file__).resolve().parent
    data_dir = final_dir / "data" / "augment"

    build_base(final_dir)
    gen_swapped(data_dir)
    gen_no_subquery(data_dir)
    gen_decompose(data_dir)


if __name__ == "__main__":
    main()