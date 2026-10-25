import ast
import re
from decimal import Decimal

DEFAULT_TOOLS = {
    "add": ["a", "b"], "subtract": ["a", "b"],
    "multiply": ["a", "b"], "divide": ["a", "b"],
    "greater": ["a","b"],
    "table_max": ["table","category","start","end"],
    "table_min": ["table","category","start","end"],
    "table_sum": ["table","category","start","end"],
    "table_average": ["table","category","start","end"],

    "chart_at": ["image", "category", "start", "end"],
    "chart_max": ["image", "category", "start", "end"],
    "chart_min": ["image", "category", "start", "end"],
    "chart_average": ["image", "category", "start", "end"],
    "chart_sum": ["image", "category", "start", "end"],
    "exp": ["a", "b"],
    "chart_total": ["image", "category", "start", "end"],
}


NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def parse_paper_plan(text):
    """Parse numbered calls without eval/exec; bridge $IDs to executor #indices."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text.endswith("<END_OF_PLAN>"):
        raise ValueError("Missing <END_OF_PLAN>")
    body = text[:-len("<END_OF_PLAN>")].strip()
    steps, ids, cursor, previous_id, joined = [], {}, 0, 0, False
    while cursor < len(body):
        match = re.match(r"\s*(\d+)\.\s*([a-zA-Z_]\w*)\s*\(", body[cursor:])
        if not match or joined:
            raise ValueError("Expected numbered tool call; join must be last")
        action_id, op = int(match[1]), match[2]
        if action_id <= previous_id or (not steps and action_id != 1):
            raise ValueError("Action IDs must start at 1 and strictly increase")
        previous_id = action_id
        start = cursor + match.start(2)
        pos = cursor + match.end()
        depth, quote, escaped = 1, None, False
        while pos < len(body) and depth:
            ch = body[pos]
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            pos += 1
        if depth:
            raise ValueError("Unclosed function call")
        try:
            call = ast.parse(body[start:pos], mode="eval").body
        except SyntaxError as exc:
            raise ValueError("Invalid call syntax") from exc
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
            raise ValueError("Expected tool(named_argument=literal)")
        kwargs = {}
        for kw in call.keywords:
            if kw.arg is None or kw.arg in kwargs:
                raise ValueError("Duplicate or unpacked keyword")
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        if op == "join":
            if kwargs or not steps:
                raise ValueError("join takes no arguments and requires a preceding result")
            joined = True
        else:
            if op not in DEFAULT_TOOLS:
                raise ValueError(f"Tool not in paper: {op}")
            expected = DEFAULT_TOOLS[op]
            optional = {"context"} if op in {"add", "subtract", "multiply", "divide"} else set()
            if not set(expected) <= kwargs.keys() or set(kwargs) - set(expected) - optional:
                raise ValueError(f"Wrong arguments for {op}")
            if "context" in kwargs and kwargs["context"] is not None and not (
                isinstance(kwargs["context"], list) and all(isinstance(x, str) for x in kwargs["context"])):
                raise ValueError("context must be a list of strings or None")
            args = []
            for name in expected:
                value = kwargs[name]
                if not isinstance(value, str):
                    raise ValueError("Paper tool arguments must be strings")
                if re.fullmatch(r"\$\d+", value):
                    ref = int(value[1:])
                    if ref not in ids:
                        raise ValueError("Invalid or forward result reference")
                    value = f"#{ids[ref]}"
                elif value.startswith("#"):
                    raise ValueError("Planner must use $id references")
                args.append(value)
            ids[action_id] = len(steps)
            steps.append({"op": op, "args": args})
        cursor = pos
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
    if not joined:
        raise ValueError("Missing final join()")
    return normalize_steps({"steps": steps}, DEFAULT_TOOLS)


def normalize_steps(obj, tools):
    steps = obj.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 64:
        raise ValueError("Expected 1-64 steps; " + str(obj.get("error", "")))
    result = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError("Each step must be an object")
        op, args = step.get("op"), step.get("args")
        if op not in tools or not isinstance(args, list) or len(args) != len(tools[op]):
            raise ValueError(f"Invalid operator or arity at step {i}")
        clean = []
        for arg in args:
            if isinstance(arg, bool) or not isinstance(arg, (str, int, float)):
                raise ValueError("Arguments must be strings or finite numbers")
            arg = str(arg).strip()
            if not arg or any(c in arg for c in "\n\r"):
                raise ValueError("Empty argument or newline in argument")
            if arg.startswith("#"):
                if not re.fullmatch(r"#\d+", arg) or int(arg[1:]) >= i:
                    raise ValueError("Invalid/forward result reference")
                arg = "#" + str(int(arg[1:]))
            elif op in {
                    "add", "subtract", "multiply",
                    "divide", "greater", "exp",
                }:
                    if not NUMBER.fullmatch(arg):
                        raise ValueError(
                            "Arithmetic arguments must use decimal-point numbers or #refs"
                        )

                    value = Decimal(arg)

                    if not value.is_finite() or abs(value) > Decimal("1e100"):
                        raise ValueError("Number outside supported range")

                    # Đổi dạng khoa học sang dạng thường nếu có.
                    # Không xóa số 0 thập phân ở literal thông thường.
                    if "e" in arg.lower():
                        arg = format(value, "f")
            clean.append(arg)
        if op == "chart_at" and clean[3] != "none":
            raise ValueError("chart_at requires end='none'")

        if op == "chart_total" and clean[2:] != ["none", "none"]:
            raise ValueError(
                "chart_total requires start='none' and end='none'"
            )
        result.append({"op": op, "args": clean})
    return result


def render_program(steps, separator=";"):
    arg_sep = separator + " "
    return "; ".join(
        f'{s["op"]}({arg_sep.join(s["args"])})'
        for s in steps
    )


def submission_to_planner(program):
    """
    Chuyển DSL submission sang planner format.

    Thành công: (plan, None)
    Không dùng được: (None, reason)

    DSL đầu vào dùng literal không bao ngoài bằng dấu nháy,
    giống qa.program trong dataset hiện tại.
    """
    try:
        if not isinstance(program, str) or not program.strip():
            return None, "empty_program"

        text = program.strip()
        if len(text) > 100000:
            return None, "program_too_long"

        calls = []
        pos = 0

        while pos < len(text):
            match = re.match(
                r"([A-Za-z_]\w*)\s*\(",
                text[pos:],
            )
            if not match:
                return None, f"invalid_call_at:{pos}"

            op = match.group(1)
            names = DEFAULT_TOOLS.get(op)

            if names is None:
                return None, f"unsupported_operator:{op}"

            pos += match.end()
            depth = 1
            start = pos
            args = []

            # Chỉ tách ; ở cấp đối số ngoài cùng.
            # Giữ ngoặc bên trong tên như P/E (x).
            while pos < len(text):
                char = text[pos]

                if char == "(":
                    depth += 1

                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        args.append(text[start:pos].strip())
                        pos += 1
                        break

                elif char == ";" and depth == 1:
                    args.append(text[start:pos].strip())
                    start = pos + 1

                pos += 1

            if depth:
                return None, "unbalanced_parentheses"

            if len(args) != len(names) or any(not a for a in args):
                return None, f"invalid_arguments:{op}"

            # Bỏ ví dụ sai contract, không tự sửa evidence.
            if op == "chart_at" and args[3] != "none":
                return None, "chart_at_requires_none"

            if op == "chart_total" and args[2:] != ["none", "none"]:
                return None, "chart_total_requires_none"

            converted = []

            for name, value in zip(names, args):
                if value.startswith("#"):
                    ref = re.fullmatch(r"#(\d+)", value)

                    if not ref or int(ref.group(1)) >= len(calls):
                        return None, f"invalid_reference:{value}"

                    value = "$" + str(int(ref.group(1)) + 1)

                elif value.startswith("$"):
                    return None, "unexpected_planner_reference"

                # repr giữ đúng dấu nháy/backslash trong literal.
                converted.append(f"{name}={value!r}")

            calls.append(
                f"{len(calls) + 1}. {op}({', '.join(converted)})"
            )

            if len(calls) > 64:
                return None, "too_many_steps"

            # Cho phép phân cách các call bằng ; hoặc xuống dòng.
            gap_start = pos

            while pos < len(text) and text[pos].isspace():
                pos += 1

            if pos < len(text):
                if text[pos] == ";":
                    pos += 1
                    while pos < len(text) and text[pos].isspace():
                        pos += 1

                elif not any(
                    c in text[gap_start:pos] for c in "\n\r"
                ):
                    return None, f"missing_call_separator:{pos}"

        plan = "\n".join(
            calls
            + [
                f"{len(calls) + 1}. join()",
                "<END_OF_PLAN>",
            ]
        )

        # Kiểm tra bằng chính parser sẽ đọc output của model.
        # Không dùng steps đã normalize để viết lại literal.
        parse_paper_plan(plan)

        return plan, None

    except Exception as exc:
        # Lỗi một ví dụ không làm hỏng inference của mẫu hiện tại.
        return None, f"{type(exc).__name__}: {exc}"