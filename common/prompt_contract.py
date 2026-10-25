import re
import json


def select_examples(examples, question, qid, adapter, top_k=5, exclude_index=None):
    selected, audit, seen = [], [], set()
    for ex in examples:
        q = ex.get("question", "")
        key = " ".join(q.split()).casefold() if isinstance(q, str) else ""
        reason = None
        ex_id = ex.get("qid", ex.get("id"))
        index = ex.get("train_index", ex.get("index"))
        if ((ex_id is not None and str(ex_id) == str(qid))
                or (exclude_index is not None and index == exclude_index)
                or key == " ".join(question.split()).casefold()):
            reason = "self_example"
        elif not key:
            reason = "empty_question"
        elif key in seen:
            reason = "duplicate_question"
        if reason is None:
            plan, reason = adapter(ex.get("program", ex.get("answer", "")))
            if plan is None:
                reason = reason or "invalid_program"
        if reason is None:
            selected.append({"question": q, "plan": plan})
            seen.add(key)
        audit.append({"index": index, "qid": ex_id,
                      "status": "skipped" if reason else "ok", "reason": reason})
        if len(selected) >= top_k:
            break
    return selected, audit


def planner_user(context_text, question, evidence, examples):
    parts = ["BỐI CẢNH HIỆN TẠI:\n" + context_text,
             "CÂU HỎI HIỆN TẠI: " + question]
    if evidence:
        lines = []
        for i, item in enumerate(evidence, 1):
            answer = item.get("answer", "")
            if not isinstance(answer, str):
                answer = json.dumps(answer, ensure_ascii=False)
            lines.append(f"Truy vấn {i}: {item.get('query', '').strip()}\nKết quả:\n{answer.strip()}")
        parts.append("BỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON:\n" + "\n\n".join(lines))
    if examples:
        parts.append("VÍ DỤ MINH HỌA:\n" + "\n\n".join(
            f"Ví dụ {i}:\nCâu hỏi: {ex['question']}\nProgram:\n{ex['plan']}"
            for i, ex in enumerate(examples, 1)))
    return "\n\n".join(parts)


def render_inline_context(context):
    """Expand each known source once; append sources absent from the text."""
    sources = {str(k): str(v) for k, v in context["table"].items()}
    for label in context["image_labels"].values():
        sources[label] = context.get("chart_perception", {}).get(label, {}).get(
            "raw_output", "Caption chưa được cung cấp.")
    lookup = {re.sub(r"\s+", "", k).casefold(): k for k in sources}
    seen = set()

    def block(label):
        seen.add(label)
        return f"### {label} ###\n{sources[label]}"

    def replace(match):
        label = lookup.get(re.sub(r"\s+", "", match[1]).casefold())
        if label is None:
            return match[0]
        return "" if label in seen else block(label)

    text = context.get("text") or []
    if isinstance(text, str):
        text = [text]
    parts = [re.sub(r"###\s*([^#\n]+?)\s*###", replace, str(t)).strip() for t in text]
    parts.extend(block(label) for label in sources if label not in seen)
    return "\n\n".join(part for part in parts if part)