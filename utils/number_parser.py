"""Natural Language Number & Mixed Numeric Expression Parser for AutoDS AI Studio.

Parses written number words, magnitude multipliers (K/M/B/T), currency-formatted numbers,
parentheses negatives, and mixed numeric expressions into floats cleanly and safely.
"""

import re
from typing import Optional

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "a": 1, "an": 1,
}

TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

SCALES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
    "trillion": 1000000000000,
}

MAGNITUDE_MULTIPLIERS = {
    "k": 1000.0,
    "m": 1000000.0,
    "b": 1000000000.0,
    "t": 1000000000000.0,
}


def parse_written_number(text: str) -> Optional[float]:
    """Parse a written natural language number expression into a float.

    Examples:
        "one hundred"                      -> 100.0
        "five hundred"                     -> 500.0
        "one thousand"                     -> 1000.0
        "one thousand two hundred fifty"  -> 1250.0
        "twenty five thousand"             -> 25000.0
        "two million"                      -> 2000000.0
        "three point five"                 -> 3.5

    Returns None if text contains non-number prose or cannot be safely parsed.
    """
    if not text or not isinstance(text, str):
        return None

    clean = text.lower().strip()
    clean = re.sub(r"[\,\-\_\;]", " ", clean).strip()

    tokens = [t for t in re.split(r"\s+", clean) if t]
    if not tokens:
        return None

    # Check token validity: every token must be a unit, ten, scale, joiner ('and'), or decimal point ('point', 'dot')
    valid_words = set(UNITS.keys()) | set(TENS.keys()) | set(SCALES.keys()) | {"and", "point", "dot"}
    has_word_token = False
    for t in tokens:
        if t in valid_words:
            if t != "and":
                has_word_token = True
            continue
        # Allow pure digits embedded in word expressions e.g. "2 million" or "1.5 thousand"
        num_cand = t.replace(".", "", 1)
        if num_cand.isdigit():
            continue
        return None  # Contains non-number prose word (e.g. "reasons", "people")

    if not has_word_token:
        return None  # Pure digit string without written number words, handled by standard numeric parser

    total_sum = 0.0
    current_section = 0.0
    in_fraction = False
    fraction_multiplier = 0.1

    for tok in tokens:
        if tok == "and":
            continue

        if tok in ("point", "dot"):
            in_fraction = True
            continue

        if in_fraction:
            val = None
            if tok in UNITS:
                val = UNITS[tok]
            elif tok.replace(".", "", 1).isdigit():
                val = float(tok)
            if val is not None and val < 10:
                current_section += val * fraction_multiplier
                fraction_multiplier *= 0.1
            else:
                return None
            continue

        if tok in UNITS:
            current_section += UNITS[tok]
        elif tok in TENS:
            current_section += TENS[tok]
        elif tok == "hundred":
            if current_section == 0:
                current_section = 1.0
            current_section *= 100.0
        elif tok in SCALES:
            if current_section == 0:
                current_section = 1.0
            current_section *= SCALES[tok]
            total_sum += current_section
            current_section = 0.0
        elif tok.replace(".", "", 1).isdigit():
            current_section += float(tok)
        else:
            return None

    final_val = total_sum + current_section
    return float(final_val)


def parse_magnitude_abbreviation(text: str) -> Optional[float]:
    """Parse magnitude abbreviated strings like '50K', '1.5M', '$50K', '2B'.

    Examples:
        "50K"     -> 50000.0
        "1.5M"    -> 1500000.0
        "$50K"    -> 50000.0
        "2B PKR"  -> 2000000000.0

    Returns None if text does not match a magnitude format.
    """
    if not text or not isinstance(text, str):
        return None

    clean = text.strip()

    # Pattern: optional currency symbol/word, number (digits/dots/commas), magnitude suffix (k/m/b/t), optional currency code
    pattern = r"^[\$\€\£\¥\₹\₦\s]*([0-9\.\,]+)\s*([kmbtKMBT])\b[\sA-Za-z]*$"
    match = re.match(pattern, clean)
    if not match:
        return None

    num_part = match.group(1).replace(",", "")
    suffix = match.group(2).lower()

    try:
        val = float(num_part)
        mult = MAGNITUDE_MULTIPLIERS.get(suffix, 1.0)
        return val * mult
    except (ValueError, TypeError):
        return None
