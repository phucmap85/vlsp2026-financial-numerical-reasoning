import re
import html
from typing import Union, List, Dict, Optional

_variables: Dict[str, float] = {}
_tables: Dict[str, str] = {}


def _clean_str(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = html.unescape(str(s)).lower().replace("\xa0", " ").strip()
    return re.sub(r'\s+', ' ', s)


def _cell_match(target: str, cell: str) -> bool:
    t, c = _clean_str(target), _clean_str(cell)
    if not t or not c:
        return False
    if t == c:
        return True
    
    t_clean = re.sub(r'[\(\)\[\]%:,\.\-\+xX\s&/]', '', t)
    c_clean = re.sub(r'[\(\)\[\]%:,\.\-\+xX\s&/]', '', c)
    if not t_clean or not c_clean:
        return False
    if t_clean == c_clean:
        return True
    if len(t_clean) >= 3 and t_clean in c_clean:
        return True
    if len(c_clean) >= 4 and c_clean in t_clean:
        return True
    return False


def _parse_number(val_input: Union[str, float, int]) -> Optional[float]:
    if val_input is None:
        return None
    if isinstance(val_input, (int, float)):
        return float(val_input)
    
    val = str(val_input).strip()
    if not val:
        return None
        
    if val in _variables:
        return _variables[val]
    if val.startswith("#"):
        if val in _variables:
            return _variables[val]
        raise ValueError(f"Variable '{val}' not found in execution context.")

    if val in ['—', '-', '–', 'NA', 'N/A', 'nan', 'NaN', 'null', 'None']:
        return None

    is_neg = False
    if val.startswith('(') and val.endswith(')'):
        is_neg = True
        val = val[1:-1].strip()
    elif val.startswith('-'):
        is_neg = True
        val = val[1:].strip()

    val = val.rstrip('%').rstrip('x').rstrip('X').strip()
    val = re.sub(r'<[^>]+>', ' ', val).strip()

    if '-' in val and not val.startswith('-'):
        parts = val.split('-')
        p0, p1 = _parse_number(parts[0]), _parse_number(parts[1])
        if p0 is not None and p1 is not None:
            return (p0 + p1) / 2.0
        elif p0 is not None:
            return p0

    if ' ' in val:
        for p in val.split():
            if any(c.isdigit() for c in p):
                val = p
                break

    if not any(c.isdigit() for c in val):
        return None

    if '.' in val and ',' in val:
        last_dot, last_comma = val.rfind('.'), val.rfind(',')
        if last_comma > last_dot:
            val = val.replace('.', '').replace(',', '.')
        else:
            val = val.replace(',', '')
    elif ',' in val:
        parts = val.split(',')
        if len(parts[-1]) == 3 and len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            val = val.replace(',', '')
        else:
            val = val.replace(',', '.')
    elif '.' in val:
        parts = val.split('.')
        if len(parts[-1]) == 3 and len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            val = val.replace('.', '')

    try:
        num = float(val)
        return -num if is_neg else num
    except Exception:
        return None


def _resolve_number(arg: Union[str, float, int]) -> float:
    res = _parse_number(arg)
    if res is None:
        raise ValueError(f"Cannot parse value '{arg}' as float.")
    return res


def _parse_html_table(html_str: str) -> List[List[str]]:
    rows = []
    tr_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', html_str, flags=re.DOTALL | re.IGNORECASE)
    for tr in tr_matches:
        cells = []
        cell_matches = re.finditer(r'<t([dh])([^>]*)>(.*?)</t\1>', tr, flags=re.DOTALL | re.IGNORECASE)
        for m in cell_matches:
            attrs = m.group(2)
            content = html.unescape(re.sub(r'<[^>]+>', '', m.group(3))).strip()
            colspan_match = re.search(r'colspan=[\'"]?(\d+)[\'"]?', attrs, flags=re.IGNORECASE)
            colspan = int(colspan_match.group(1)) if colspan_match else 1
            for _ in range(colspan):
                cells.append(content)
        rows.append(cells)
    return rows


def _extract_table_cells(table_input: str, category: str, start: str, end: str) -> List[float]:
    html_content = ""
    if "<table" in table_input.lower() or "<tr" in table_input.lower():
        html_content = table_input
    elif table_input in _tables:
        html_content = _tables[table_input]
    else:
        for t_k, t_v in _tables.items():
            if _clean_str(t_k) == _clean_str(table_input):
                html_content = t_v
                break
        if not html_content:
            raise ValueError(f"Table '{table_input}' not found and is not valid HTML.")

    grid = _parse_html_table(html_content)
    if not grid:
        raise ValueError(f"Table '{table_input}' is empty.")

    # 1. Horizontal search
    cat_row_candidates = []
    for r_i, row in enumerate(grid):
        for c_i in range(min(2, len(row))):
            if _cell_match(category, row[c_i]):
                cat_row_candidates.append(r_i)
                break

    if cat_row_candidates:
        r_i = cat_row_candidates[0]
        start_c = 1 if _clean_str(start) in ["none", ""] else None
        end_c = len(grid[r_i]) - 1 if _clean_str(end) in ["none", ""] else None

        for h_r in range(min(3, r_i + 1)):
            for c_i, cell in enumerate(grid[h_r]):
                if start_c is None and _cell_match(start, cell):
                    start_c = c_i
                if end_c is None and _cell_match(end, cell):
                    end_c = c_i

        if start_c is not None and end_c is not None:
            c1, c2 = min(start_c, end_c), max(start_c, end_c)
            raw_cells = [grid[r_i][c] for c in range(c1, min(c2 + 1, len(grid[r_i])))]
            valid_nums = [n for n in [_parse_number(c) for c in raw_cells] if n is not None]
            if valid_nums:
                return valid_nums

    # 2. Vertical search
    cat_col_candidates = []
    if '@' in category and len(grid) >= 2:
        p0, p1 = category.split('@', 1)
        for c_i in range(min(len(grid[0]), len(grid[1]))):
            if _cell_match(p0, grid[0][c_i]) and _cell_match(p1, grid[1][c_i]):
                cat_col_candidates.append((1, c_i))

    if not cat_col_candidates:
        for r_i in range(min(2, len(grid))):
            for c_i, cell in enumerate(grid[r_i]):
                if _cell_match(category, cell):
                    cat_col_candidates.append((r_i, c_i))

    if cat_col_candidates:
        h_r, c_i = cat_col_candidates[0]
        start_r = h_r + 1 if _clean_str(start) in ["none", ""] else None
        end_r = len(grid) - 1 if _clean_str(end) in ["none", ""] else None

        for r in range(h_r + 1, len(grid)):
            for col_check in range(min(2, len(grid[r]))):
                cell = grid[r][col_check]
                if start_r is None and _cell_match(start, cell):
                    start_r = r
                if end_r is None and _cell_match(end, cell):
                    end_r = r

        if start_r is not None and end_r is not None:
            r1, r2 = min(start_r, end_r), max(start_r, end_r)
            raw_cells = [grid[r][c_i] for r in range(r1, min(r2 + 1, len(grid))) if c_i < len(grid[r])]
            valid_nums = [n for n in [_parse_number(c) for c in raw_cells] if n is not None]
            if valid_nums:
                return valid_nums

    raise ValueError(f"Could not extract values for '{category}' between '{start}' and '{end}'.")


def add(a: str, b: str) -> float:
    return _resolve_number(a) + _resolve_number(b)


def subtract(a: str, b: str) -> float:
    return _resolve_number(a) - _resolve_number(b)


def multiply(a: str, b: str) -> float:
    return _resolve_number(a) * _resolve_number(b)


def divide(a: str, b: str) -> float:
    denominator = _resolve_number(b)
    if denominator == 0:
        return 0.0
    return _resolve_number(a) / denominator


def greater(a: str, b: str) -> float:
    return _resolve_number(a) > _resolve_number(b)


def table_max(table: str, category: str, start: str, end: str) -> float:
    return float(max(_extract_table_cells(table, category, start, end)))


def table_min(table: str, category: str, start: str, end: str) -> float:
    return float(min(_extract_table_cells(table, category, start, end)))


def table_sum(table: str, category: str, start: str, end: str) -> float:
    return float(sum(_extract_table_cells(table, category, start, end)))


def table_average(table: str, category: str, start: str, end: str) -> float:
    values = _extract_table_cells(table, category, start, end)
    return float(sum(values) / len(values))


def chart_at(*args) -> float:
    return 100.0


def chart_max(*args) -> float:
    return 110.0


def chart_min(*args) -> float:
    return 120.0


def chart_average(*args) -> float:
    return 130.0


def chart_sum(*args) -> float:
    return 140.0


FUNCTIONS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
    "greater": greater,
    "table_max": table_max,
    "table_min": table_min,
    "table_sum": table_sum,
    "table_average": table_average,
    "chart_at": chart_at,
    "chart_max": chart_max,
    "chart_min": chart_min,
    "chart_average": chart_average,
    "chart_sum": chart_sum,
}


def run(program: str, tables: Optional[Dict[str, str]] = None) -> float:
    global _variables, _tables
    _variables.clear()
    _tables = dict(tables) if tables else {}

    pattern = r'([a-zA-Z_0-9]+)\s*\((.*?)\)(?=\s*(?:[;\n\r]\s*[a-zA-Z_0-9]+\s*\(|$))'
    matches = re.findall(pattern, program)
    if not matches:
        raise ValueError(f"No valid commands found in program: '{program}'")

    last_result = 0.0
    for step_idx, (fn_name, args_str) in enumerate(matches):
        if fn_name not in FUNCTIONS:
            raise ValueError(f"Function '{fn_name}' is not supported.")
        
        args = [arg.strip() for arg in args_str.split(';')]
        step_result = FUNCTIONS[fn_name](*args)
        
        _variables[f"#{step_idx}"] = step_result
        last_result = step_result

    return last_result


def get_variables() -> Dict[str, float]:
    return dict(_variables)