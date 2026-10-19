"""规则式扰动：typo、数字格式、LaTeX 写法。

只用规则式变换，保证数学内容与标签不变；不用 LLM 改写
（若将来加 LLM 改写，必须验证标签不变并按 §4.8 披露）。
"""
import re


def typo(text: str, seed: int = 0, p: float = 0.10) -> str:
    import random
    rng = random.Random(seed)
    words = text.split(" ")
    out = []
    for w in words:
        if len(w) >= 4 and w.isalpha() and rng.random() < p:
            i = rng.randrange(1, len(w) - 1)
            op = rng.choice(["swap", "drop", "dup"])
            if op == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == "drop":
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


_NUM_PATTERNS = [
    (re.compile(r"(\d),(\d{3})"), r"\1\2"),          # 1,000 -> 1000
    (re.compile(r"(?<![\d.])0\.(\d)"), r".\1"),        # 0.5 -> .5
    (re.compile(r"(?<![\d.])(\.(\d))"), r"0\1"),       # .5 -> 0.5（与上一条方向互斥，靠顺序）
]


def number_format(text: str) -> str:
    t = re.sub(r"(\d),(\d{3})", r"\1\2", text)          # 去千分位
    t = re.sub(r"(?<![\d.])0\.(\d)", r".\1", t)         # 0.5 -> .5
    t = re.sub(r"(\d+)\.0+(?!\d)", r"\1", t)            # 12.0 -> 12
    return t


_LATEX_SUBS = [
    (r"\\left\(", "("), (r"\\left\[", "["), (r"\\right\)", ")"), (r"\\right\]", "]"),
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"),
    (r"\\times", r"\\cdot"), (r"\\!", ""), (r"\\,", " "), (r"\\;", " "),
]


def latex_style(text: str) -> str:
    t = text
    for pat, rep in _LATEX_SUBS:
        t = re.sub(pat, rep, t)
    # \frac{a}{b} -> a/b（只对短的原子表达式做这个等价改写）
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", t)
    return t


PERTURBATIONS = {
    "none": lambda s, seed=0: s,
    "typo": lambda s, seed=0: typo(s, seed),
    "number": lambda s, seed=0: number_format(s),
    "latex": lambda s, seed=0: latex_style(s),
    "combo": lambda s, seed=0: latex_style(number_format(typo(s, seed))),
}
