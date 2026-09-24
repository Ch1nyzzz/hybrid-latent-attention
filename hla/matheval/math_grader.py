"""Answer extraction and equivalence for \\boxed{} math answers (Hendrycks-style normalization + sympy fallback)."""
import re

def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        m = re.findall(r"[Ff]inal [Aa]nswer[^0-9\-]*(-?\d+)", text)
        return m[-1] if m else None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{": depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None

def _fix_fracs(s):
    out, parts = "", s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        if p.startswith("{"):
            out += "\\frac" + p
        elif len(p) >= 2:
            a, b, rest = p[0], p[1], p[2:]
            if b == "{":  # \frac a{...}
                out += "\\frac{" + a + "}" + p[1:]
            else:
                out += "\\frac{" + a + "}{" + b + "}" + rest
        else:
            out += "\\frac" + p
    return out

def normalize(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\n", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\\\", "\\").replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mbox|textbf)\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "")
    s = re.sub(r"^[a-zA-Z]=", "", s) if s.count("=") == 1 and len(s) > 2 and s[1] == "=" else s
    s = s.rstrip(".")
    s = s.replace("\\dfrac", "\\frac")
    s = _fix_fracs(s)
    s = re.sub(r"(\d),(\d\d\d)(?!\d)", r"\1\2", s) if "," in s and not re.search(r",\\|,\(", s) else s
    if s.startswith("0."):
        pass
    elif s.startswith("."):
        s = "0" + s
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"\\frac\{-?\d+\}\{-?\d+\}", s) or re.fullmatch(r"-?\d+/-?\d+", s):
        pass
    return s

def _latex_to_py(s):
    for _ in range(4):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    s = re.sub(r"(\d+)!", r"factorial(\1)", s)
    s = re.sub(r"\{|\}", "", s)
    s = re.sub(r"(\d)(pi|sqrt|\()", r"\1*\2", s)
    s = re.sub(r"\)(\d|\()", r")*\1", s)
    return s

def _to_sympy(s):
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(s)
    except Exception:
        pass
    try:
        import sympy
        return sympy.sympify(_latex_to_py(s))
    except Exception:
        return None

def is_equiv(pred, gold):
    if pred is None:
        return False
    a, b = normalize(pred), normalize(gold)
    if a == b:
        return True
    if a is None or b is None:
        return False
    # integer answers (AIME): exact after stripping leading zeros
    if re.fullmatch(r"-?\d+", a) and re.fullmatch(r"-?\d+", b):
        return int(a) == int(b)
    # a/b vs \frac{a}{b}
    fa = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", a); fb = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", b)
    ra = re.fullmatch(r"(-?\d+)/(-?\d+)", a); rb = re.fullmatch(r"(-?\d+)/(-?\d+)", b)
    if (fa or ra) and (fb or rb):
        na, da = (fa or ra).groups(); nb, db = (fb or rb).groups()
        return int(na) * int(db) == int(nb) * int(da)
    # symbolic / numeric fallback
    xa, xb = _to_sympy(a), _to_sympy(b)
    if xa is not None and xb is not None:
        try:
            import sympy
            d = sympy.simplify(xa - xb)
            if d == 0:
                return True
            fa_, fb_ = float(xa.evalf()), float(xb.evalf())
            return abs(fa_ - fb_) <= 1e-6 * max(1.0, abs(fb_))
        except Exception:
            return False
    return False
