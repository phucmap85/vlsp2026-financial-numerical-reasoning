import json
import re
from pathlib import Path

import pipeline


def read_rows(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(text)
    if isinstance(rows, dict):
        rows = rows['root']
    if not isinstance(rows, list):
        raise ValueError('Expected a list of samples')
    return rows



def choose(outputs, context):
    candidates = []
    for i, output in enumerate(outputs):
        candidate = dict(output, index=i)
        candidates.append(candidate)
        try:
            if output.get('finish_reason') != 'stop' or output.get('thinking_complete') is False:
                raise ValueError('Incomplete generation or thinking')
            steps = pipeline.parse_paper_plan(output.get('text') or '')
            error = pipeline.validate_source_ids(steps, context)
            if error:
                raise ValueError(error)
            candidate['steps'] = steps
        except (ValueError, TypeError, KeyError) as exc:
            candidate['error'] = str(exc)
    try:
        steps, votes = pipeline.select_program(candidates)
        program = pipeline.render_program(steps)
    except ValueError:
        program, votes = '', 0
    return program, candidates, votes


def strict_match(predicted, gold):
    # Formatting whitespace only: do not equate 9.0 with 9 or reorder operations.
    return re.sub(r'\s+', '', predicted or '') == re.sub(r'\s+', '', gold or '')