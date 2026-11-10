"""Answer grading by answer type.

The type is fixed per problem from the dataset and the ground truth, never from the prediction:
- choice: yes/no/true/false, every GQA answer (scene words), geometry3k option letters. Case, spaces, punctuation and a
  leading article (the/a/an) are ignored; synonyms are not accepted.
- number: one number, optionally with a currency sign, %, a unit word or a trailing '.' (also a number word one-ten as
  the ground truth). Equal values written differently match (separators, trailing zeros, e-notation), and so
  do a unit on one side only and the same unit abbreviated ('45 grams' = '45g'). Rounding, scale conversion ('million'
  vs 'billion') and unit conversion never match.
- text: everything else read from the image (names, titles, dates, IDs, ranges, ratios, lists): exact match, so a
  case-only difference is wrong.

A prediction that the old grading (``filter_out.answers_match``) accepted stays correct; these rules only add matches.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

UNIT_ALIASES = {"grams": "g", "gram": "g", "million": "m", "mn": "m", "mil": "m", "billion": "bn", "b": "bn",
                "thousand": "k", "percent": "%", "usd": "$", "dollars": "$", "dollar": "$", "usdollars": "$",
                "usdollar": "$", "us$": "$"}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUMBER_RE = re.compile(r"^([$€£¥]?)\s*([+-]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?(?:e[+-]?\d+)?)\s*(.*?)\.?$", re.I)


def _detex(s: str) -> str:
    s = re.sub(r"\\text\{([^}]*)\}", r" \1", s)
    return s.replace("\\%", "%").replace("\\,", " ").replace("\\ ", " ").replace("\\", "").strip()


def number(s: str) -> tuple[Decimal, str] | None:
    """(value, unit text) of a single-number answer or a number word, else None."""
    word = s.strip().rstrip(".").lower()
    if word in NUMBER_WORDS:
        return Decimal(NUMBER_WORDS[word]), ""
    m = NUMBER_RE.match(_detex(s))
    if not m:
        return None
    currency, digits, unit = m.groups()
    if re.search(r"\d", unit) or len(unit.split()) > 4:
        return None
    try:
        value = Decimal(digits.replace(",", "").replace(" ", ""))
    except InvalidOperation:
        return None
    return value, re.sub(r"[\s.]", "", (currency + unit).lower()).replace("u.s.", "us")


def _unit(u: str) -> str:
    u = u.replace("us", "") if u.endswith("dollars") or u.endswith("dollar") else u
    return UNIT_ALIASES.get(u, u)


def answer_type(dataset: str, answer: str) -> str:
    """choice, number or text for a problem's ground truth."""
    g = answer.strip().lower().rstrip(".")
    if g in ("yes", "no", "true", "false") or dataset.startswith("gqa") or dataset == "geometry3k":
        return "choice"
    return "number" if number(answer) else "text"


def _choice_key(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", re.sub(r"^\s*(the|a|an)\s+", "", s.strip().lower()))


def matches(prediction: str, answer: str, kind: str) -> bool:
    """Whether the type rule accepts ``prediction`` (the stored boxed answer) for ``answer``."""
    prediction = prediction.strip()
    if kind == "choice":
        return bool(_choice_key(prediction)) and _choice_key(prediction) == _choice_key(answer)
    if kind == "text":
        return False  # exact match only, which the old grading already decides
    a, b = number(prediction), number(answer)
    if not a or not b or a[0] != b[0]:
        return False
    return a[1] == b[1] or a[1] == "" or b[1] == "" or _unit(a[1]) == _unit(b[1])


def grade(prediction: str | None, answer: str, dataset: str, strict_correct: bool) -> bool:
    """Correct under the type rules: the old grading's verdict, or a type-rule match."""
    return strict_correct or (bool(prediction) and matches(prediction, answer, answer_type(dataset, answer)))
