"""Generalized semantic categorical value standardization and normalization engine.

Upgrades:
1. Purely algorithmic candidate generation — ZERO hardcoded domain/city/date/acronym tables.
2. Distinguishes:
   - FORMAT DIFFERENCE (casing, whitespace, Unicode, punctuation) -> Safe (0.95 - 1.00)
   - TYPOGRAPHICAL (edit distance with strong frequency asymmetry) -> High (0.85 - 0.94)
   - SEMANTIC / ABBREVIATION (co-occurring acronyms and prefixes) -> High/Review
   - ACTUALLY DIFFERENT VALUES (preserves distinct concepts, avoids false merges)
3. Context-Aware Evaluation: Checks co-occurrence in other categorical columns.
4. Ambiguity Guard: Flags abbreviations with multiple potential meanings for user review.
5. Evidence-based canonical value selection: Prioritizes full descriptive terms, high frequency, and Title Case.
6. Categorical vs Free Text Guard: Protects long text, comments, descriptions, and identifiers.
"""

import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Any, Optional
import pandas as pd

from core.logging import logger
from models.repair import IssueTaxonomy, RepairRecord


# Generic null sentinel representations across tabular data
GENERIC_NULL_SENTINELS = {
    "",
    "null",
    "none",
    "nan",
    "na",
    "n/a",
    "n.a.",
    "n/d",
    "-",
    "--",
    "---",
    "?",
    "??",
    "???",
    "????",
    ".",
    "..",
    "...",
    "unknown",
    "missing",
    "#n/a",
    "#na",
    "#null!",
    "undefined",
    "nil",
    "blank",
}

# Distinct entity qualifiers that indicate separate business or operational concepts, NOT typos
DISTINCT_ENTITY_QUALIFIERS = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation", "co", "company",
    "store", "shop", "center", "centre", "mall", "outlet", "market", "mart",
    "plus", "pro", "max", "mini", "lite", "ultra", "standard", "premium",
    "north", "south", "east", "west", "central", "hq", "branch",
    "group", "division", "dept", "department", "service", "services", "solutions",
}


def _is_distinct_qualified_entity(s1: str, s2: str) -> bool:
    """Guard against merging distinct entities that happen to share word roots or prefixes.

    Examples that must NEVER be merged:
    - 'ABC' vs 'ABC Ltd' vs 'ABC Store'
    - 'Apple' vs 'Apple Store'
    - 'Standard' vs 'Standard Plus'
    """
    w1 = set(re.findall(r"\b[a-z0-9]+\b", clean_unicode_and_whitespace(s1).lower()))
    w2 = set(re.findall(r"\b[a-z0-9]+\b", clean_unicode_and_whitespace(s2).lower()))

    if not w1 or not w2:
        return False

    diff = (w1 ^ w2)
    # If the differing words include any entity/qualifier word, they are distinct entities!
    if any(d in DISTINCT_ENTITY_QUALIFIERS for d in diff):
        return True

    # If one is a single short word and the other is a multi-word phrase containing it plus another content word
    if (len(w1) == 1 and len(w2) > 1) or (len(w2) == 1 and len(w1) > 1):
        if w1.issubset(w2) or w2.issubset(w1):
            return True

    return False


# Maximum unique values in a column to attempt categorical clustering
MAX_CARDINALITY_THRESHOLD = 250


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def clean_unicode_and_whitespace(value: Any) -> str:
    """Normalize Unicode, strip invisible characters, and collapse spaces."""
    if not isinstance(value, str):
        return f"{value}"
    # NFKC normalizes compatibility characters & ligatures
    s = unicodedata.normalize("NFKC", value)
    # Strip zero-width & non-breaking spaces
    s = s.replace("\u200b", "").replace("\ufeff", "").replace("\u200e", "").replace("\u00a0", " ")
    # Collapse multiple spaces, tabs, newlines
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_canonical_key(value: str) -> str:
    """Compute base normalization key for case & punctuation clustering."""
    s = clean_unicode_and_whitespace(value).lower()
    # Remove all non-alphanumeric characters for key comparison
    return re.sub(r"[^a-z0-9]", "", s)


def extract_token_key(value: str) -> tuple[str, ...]:
    """Extract sorted tokens for permutation matching (e.g. 'Credit Card' vs 'Card, Credit')."""
    s = clean_unicode_and_whitespace(value).lower()
    tokens = tuple(sorted(re.findall(r"\b[a-z0-9]+\b", s)))
    return tokens


def is_categorical_column(series: pd.Series, max_cardinality: int = MAX_CARDINALITY_THRESHOLD) -> bool:
    """Determine whether a text series is a categorical column or free-form text.

    Protects:
    - Free-form text (long descriptions, comments, free responses)
    - Full names, addresses
    - Identifiers / unique codes
    """
    if series.dtype != "object" and not pd.api.types.is_string_dtype(series):
        return False

    non_null = series.dropna()
    total = len(non_null)
    if total == 0:
        return False

    unique_count = non_null.nunique()
    if unique_count <= 1 or unique_count > max_cardinality:
        return False

    # Check cardinality ratio: categoricals typically have repeated entries
    if total > 20 and (unique_count / total) > 0.85:
        return False

    str_sample = non_null.astype(str).head(100)

    # Check average string length and word count
    avg_len = str_sample.str.len().mean()
    if avg_len > 60:
        return False

    avg_words = str_sample.str.split().str.len().mean()
    if avg_words > 8:
        return False

    return True


def evaluate_context_correlation(
    df: pd.DataFrame, target_col: str, val1: str, val2: str
) -> float:
    """Evaluate whether val1 and val2 co-occur with consistent context in other columns.

    Returns a correlation modifier between 0.0 (conflicting) and 1.0 (strong contextual agreement).
    """
    context_cols = [
        c for c in df.columns
        if c != target_col
        and (df[c].dtype == "object" or isinstance(df[c].dtype, pd.CategoricalDtype))
        and df[c].nunique() <= 50
    ]

    if not context_cols:
        return 0.5  # Neutral if no context columns available

    agreement_scores = []
    for ctx_col in context_cols:
        ctx1 = set(df[df[target_col] == val1][ctx_col].dropna().unique())
        ctx2 = set(df[df[target_col] == val2][ctx_col].dropna().unique())

        if not ctx1 or not ctx2:
            continue

        intersection = ctx1 & ctx2
        union = ctx1 | ctx2
        if union:
            agreement_scores.append(len(intersection) / len(union))

    if not agreement_scores:
        return 0.5

    return sum(agreement_scores) / len(agreement_scores)


def select_canonical_value(candidates: list[str], val_counts: dict[str, int]) -> str:
    """Select the best canonical representation from a group of candidate values.

    Rules:
    1. Prefer higher frequency in dataset
    2. Prefer Title Case or UPPERCASE over lowercase
    3. Prefer longer descriptive phrase over single words
    """
    def score_candidate(v: str) -> tuple[int, int, int]:
        freq = val_counts.get(v, 0)
        is_cased = 2 if v.istitle() else (1 if (v.isupper() or any(c.isupper() for c in v)) else 0)
        word_count = len(v.split())
        return (freq, is_cased, word_count)

    sorted_candidates = sorted(candidates, key=score_candidate, reverse=True)
    return sorted_candidates[0]


def get_proposed_standardizations(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    max_cardinality: int = MAX_CARDINALITY_THRESHOLD,
    extra_abbreviations: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Analyze DataFrame and return validated standardization proposals across categorical columns.

    Zero hardcoded domain rules. Distinguishes:
    - Null sentinels (NA, NULL, etc.)
    - Case, whitespace, & unicode formatting variants
    - Typographical variants with frequency asymmetry
    - Token-set permutations
    - Dynamic co-occurring acronyms and prefix abbreviations
    - Ambiguous abbreviations flagged for review
    """
    proposals: list[dict[str, Any]] = []

    target_cols = columns if columns else [
        c for c in df.columns if is_categorical_column(df[c], max_cardinality=max_cardinality)
    ]

    for col in target_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        val_counts = series.astype(str).value_counts().to_dict()
        unique_vals = list(val_counts.keys())
        if len(unique_vals) <= 1:
            continue

        # ── 1. Null Sentinel Detection ──
        for val in unique_vals:
            cleaned = clean_unicode_and_whitespace(val).lower()
            if cleaned in GENERIC_NULL_SENTINELS:
                proposals.append({
                    "column": col,
                    "original_value": val,
                    "normalized_value": None,
                    "confidence": 0.99,
                    "method": "null_sentinel",
                    "issue_type": IssueTaxonomy.MISSING.value,
                    "reason": f"Recognized generic missing data marker '{val}'",
                    "affected_rows": val_counts[val],
                    "is_safe": True,
                })

        active_vals = [
            v for v in unique_vals
            if clean_unicode_and_whitespace(v).lower() not in GENERIC_NULL_SENTINELS
        ]
        if not active_vals:
            continue

        # ── 2. Format Inconsistencies: Case, Whitespace & Punctuation ──
        key_to_vals: dict[str, list[str]] = {}
        for val in active_vals:
            key = extract_canonical_key(val)
            if key:
                if key not in key_to_vals:
                    key_to_vals[key] = []
                key_to_vals[key].append(val)

        canonical_by_key: dict[str, str] = {}
        for key, vals in key_to_vals.items():
            if len(vals) > 1:
                canonical = select_canonical_value(vals, val_counts)
                canonical_by_key[key] = canonical

                for v in vals:
                    if v != canonical:
                        is_pure_ws = clean_unicode_and_whitespace(v).lower() == clean_unicode_and_whitespace(canonical).lower()
                        confidence = 0.98 if is_pure_ws else 0.95
                        method = "whitespace_case_normalization" if is_pure_ws else "punctuation_normalization"
                        proposals.append({
                            "column": col,
                            "original_value": v,
                            "normalized_value": canonical,
                            "confidence": confidence,
                            "method": method,
                            "issue_type": IssueTaxonomy.FORMAT.value,
                            "reason": f"Format standardization '{v}' → '{canonical}'",
                            "affected_rows": val_counts.get(v, 0),
                            "is_safe": True,
                        })
            else:
                canonical_by_key[key] = vals[0]

        # ── 3. Token-set Permutation (e.g. 'Card, Credit' vs 'Credit Card') ──
        token_to_canonicals: dict[tuple[str, ...], list[str]] = {}
        for key, canonical in canonical_by_key.items():
            token_key = extract_token_key(canonical)
            if len(token_key) >= 2:
                if token_key not in token_to_canonicals:
                    token_to_canonicals[token_key] = []
                token_to_canonicals[token_key].append(canonical)

        for token_key, group in token_to_canonicals.items():
            if len(group) > 1:
                dominant_canonical = select_canonical_value(group, val_counts)
                for other_canon in group:
                    if other_canon != dominant_canonical:
                        ctx_score = evaluate_context_correlation(df, col, other_canon, dominant_canonical)
                        confidence = round(0.88 + (ctx_score * 0.05), 2)
                        proposals.append({
                            "column": col,
                            "original_value": other_canon,
                            "normalized_value": dominant_canonical,
                            "confidence": confidence,
                            "method": "token_permutation",
                            "issue_type": IssueTaxonomy.FORMAT.value,
                            "reason": f"Word order permutation variant '{other_canon}' → '{dominant_canonical}'",
                            "affected_rows": val_counts.get(other_canon, 0),
                            "is_safe": confidence >= 0.90,
                        })

        # ── 4. Typographical Fuzzy Clustering with Frequency Asymmetry ──
        canonical_list = list(canonical_by_key.values())
        seen_fuzzy_pairs: set[tuple[str, str]] = set()

        for i in range(len(canonical_list)):
            for j in range(i + 1, len(canonical_list)):
                c1 = canonical_list[i]
                c2 = canonical_list[j]

                # Rule 2: NEVER assume string similarity = same meaning
                if _is_distinct_qualified_entity(c1, c2):
                    continue

                k1 = extract_canonical_key(c1)
                k2 = extract_canonical_key(c2)

                if len(k1) < 4 or len(k2) < 4:
                    continue  # Protect short codes and abbreviations

                dist = levenshtein_distance(k1, k2)
                max_len = max(len(k1), len(k2))
                sim = 1.0 - (dist / max_len)

                # Strict typo threshold: edit distance 1 (>=75%) or 2 (>=75% with length >=7)
                is_candidate_typo = (dist == 1 and sim >= 0.75) or (dist == 2 and sim >= 0.75 and max_len >= 7)
                if not is_candidate_typo:
                    continue

                cnt1 = sum(val_counts.get(v, 0) for v in key_to_vals.get(k1, [c1]))
                cnt2 = sum(val_counts.get(v, 0) for v in key_to_vals.get(k2, [c2]))

                # Frequency asymmetry is critical: prevents merging legitimately distinct concepts
                if cnt1 >= 3 * cnt2 or (cnt1 >= 4 and cnt2 <= 1):
                    dominant, variant, dom_cnt, var_cnt = c1, c2, cnt1, cnt2
                    is_safe = True
                    confidence = round(min(0.95, 0.86 + (sim * 0.08)), 2)
                elif cnt2 >= 3 * cnt1 or (cnt2 >= 4 and cnt1 <= 1):
                    dominant, variant, dom_cnt, var_cnt = c2, c1, cnt2, cnt1
                    is_safe = True
                    confidence = round(min(0.95, 0.86 + (sim * 0.08)), 2)
                else:
                    # Balanced frequency: requires user review
                    dominant = c1 if cnt1 >= cnt2 else c2
                    variant = c2 if cnt1 >= cnt2 else c1
                    is_safe = False
                    confidence = 0.75

                pair_key = (variant, dominant)
                if pair_key not in seen_fuzzy_pairs:
                    seen_fuzzy_pairs.add(pair_key)
                    proposals.append({
                        "column": col,
                        "original_value": variant,
                        "normalized_value": dominant,
                        "confidence": confidence,
                        "method": "fuzzy_typo",
                        "issue_type": IssueTaxonomy.TYPOGRAPHICAL.value,
                        "reason": f"Likely typographical variant of '{dominant}' ({sim:.0%} similarity, freq ratio {max(cnt1, cnt2)}:{min(cnt1, cnt2)})",
                        "affected_rows": val_counts.get(variant, 0),
                        "is_safe": is_safe,
                    })

        # ── 5. Dynamic Acronym & Abbreviation Discovery ──
        multi_word_canonicals = [c for c in canonical_list if " " in clean_unicode_and_whitespace(c)]
        short_tokens = [c for c in canonical_list if 2 <= len(clean_unicode_and_whitespace(c)) <= 10 and " " not in clean_unicode_and_whitespace(c)]
        longer_words = [c for c in canonical_list if len(clean_unicode_and_whitespace(c)) >= 4]

        # 5a. Acronym matching initials of multi-word phrases (e.g. 'IBM' vs 'International Business Machines')
        acronym_candidates: dict[str, list[str]] = {}
        for acr in short_tokens:
            if len(clean_unicode_and_whitespace(acr)) > 4:
                continue
            acr_clean = clean_unicode_and_whitespace(acr).upper()
            for mw in multi_word_canonicals:
                tokens = [t for t in re.findall(r"\b[A-Za-z0-9]+", mw)]
                initials_all = "".join(t[0] for t in tokens).upper()
                content_tokens = [
                    t for t in tokens
                    if t.lower() not in {"of", "the", "and", "in", "for", "on", "at", "to", "a", "an"}
                ]
                initials_content = "".join(t[0] for t in content_tokens).upper()

                if acr_clean in {initials_all, initials_content} and len(acr_clean) >= 2:
                    if acr not in acronym_candidates:
                        acronym_candidates[acr] = []
                    if mw not in acronym_candidates[acr]:
                        acronym_candidates[acr].append(mw)

        # 5b. Prefix Abbreviation / Truncation (e.g. 'Furn' vs 'Furniture', 'App' vs 'APPAREL')
        prefix_candidates: dict[str, list[str]] = {}
        for short_tok in short_tokens:
            short_clean = clean_unicode_and_whitespace(short_tok).lower()
            if len(short_clean) < 3 or len(short_clean) > 5:
                continue
            # Skip if already identified as a multi-word acronym
            if short_tok in acronym_candidates:
                continue

            for lw in longer_words:
                lw_clean = clean_unicode_and_whitespace(lw).lower()
                if len(lw_clean) >= len(short_clean) + 2 and lw_clean.startswith(short_clean):
                    if not _is_distinct_qualified_entity(short_tok, lw):
                        if short_tok not in prefix_candidates:
                            prefix_candidates[short_tok] = []
                        if lw not in prefix_candidates[short_tok]:
                            prefix_candidates[short_tok].append(lw)

        # 5c. Consonant Skeleton / Syllable Subsequence (e.g. 'Lhr' vs 'Lahore', 'ISB' vs 'Islamabad')
        consonant_candidates: dict[str, list[str]] = {}
        vowels = set("aeiou")
        for short_tok in short_tokens:
            if short_tok in acronym_candidates or short_tok in prefix_candidates:
                continue
            s_clean = clean_unicode_and_whitespace(short_tok).lower()
            if len(s_clean) < 2 or len(s_clean) > 4:
                continue

            for lw in longer_words:
                lw_clean = clean_unicode_and_whitespace(lw).lower()
                if len(lw_clean) <= len(s_clean) + 2:
                    continue
                # Must start with the same initial letter
                if s_clean[0] != lw_clean[0]:
                    continue

                # Test if s_clean characters appear as an ordered subsequence in lw_clean
                it = iter(lw_clean)
                is_subseq = all(c in it for c in s_clean)
                if is_subseq:
                    # Verify that the intermediate characters matched are predominantly consonants
                    lw_consonants = "".join(c for c in lw_clean if c not in vowels)
                    s_consonants = "".join(c for c in s_clean if c not in vowels)
                    it_cons = iter(lw_consonants)
                    is_cons_subseq = all(c in it_cons for c in s_consonants) if s_consonants else True

                    if is_cons_subseq and not _is_distinct_qualified_entity(short_tok, lw):
                        if short_tok not in consonant_candidates:
                            consonant_candidates[short_tok] = []
                        if lw not in consonant_candidates[short_tok]:
                            consonant_candidates[short_tok].append(lw)

        # Emit acronym proposals
        for acr, matches in acronym_candidates.items():
            if len(matches) == 1:
                target_full = matches[0]
                ctx_score = evaluate_context_correlation(df, col, acr, target_full)
                conf = round(0.88 + (ctx_score * 0.05), 2)
                proposals.append({
                    "column": col,
                    "original_value": acr,
                    "normalized_value": target_full,
                    "confidence": conf,
                    "method": "acronym_expansion",
                    "issue_type": IssueTaxonomy.SEMANTIC.value,
                    "reason": f"Co-occurring acronym '{acr}' matches initials of '{target_full}'",
                    "affected_rows": val_counts.get(acr, 0),
                    "is_safe": True,
                    "is_ambiguous": False,
                })
            else:
                proposals.append({
                    "column": col,
                    "original_value": acr,
                    "normalized_value": matches[0],
                    "confidence": 0.72,
                    "method": "ambiguous_abbreviation",
                    "issue_type": IssueTaxonomy.SEMANTIC.value,
                    "reason": f"Ambiguous acronym '{acr}' matches multiple terms: {', '.join(matches)}",
                    "affected_rows": val_counts.get(acr, 0),
                    "is_safe": False,
                    "is_ambiguous": True,
                    "possible_meanings": matches,
                    "candidate_meanings": matches,
                })

        # Emit prefix proposals
        for pfx, matches in prefix_candidates.items():
            if len(matches) == 1:
                target_full = matches[0]
                cnt_pfx = val_counts.get(pfx, 0)
                cnt_full = val_counts.get(target_full, 0)
                ctx_score = evaluate_context_correlation(df, col, pfx, target_full)
                conf = round(min(0.90, 0.82 + (0.05 if cnt_full >= cnt_pfx else 0.0) + (ctx_score * 0.04)), 2)
                proposals.append({
                    "column": col,
                    "original_value": pfx,
                    "normalized_value": target_full,
                    "confidence": conf,
                    "method": "prefix_abbreviation",
                    "issue_type": IssueTaxonomy.SEMANTIC.value,
                    "reason": f"Prefix abbreviation '{pfx}' dynamically matches canonical word '{target_full}'",
                    "affected_rows": cnt_pfx,
                    "is_safe": conf >= 0.85,
                    "is_ambiguous": False,
                })
            else:
                proposals.append({
                    "column": col,
                    "original_value": pfx,
                    "normalized_value": matches[0],
                    "confidence": 0.70,
                    "method": "ambiguous_prefix_abbreviation",
                    "issue_type": IssueTaxonomy.SEMANTIC.value,
                    "reason": f"Ambiguous prefix '{pfx}' matches multiple candidate terms: {', '.join(matches)}",
                    "affected_rows": val_counts.get(pfx, 0),
                    "is_safe": False,
                    "is_ambiguous": True,
                    "possible_meanings": matches,
                    "candidate_meanings": matches,
                })

        # Emit consonant skeleton proposals
        for skel, matches in consonant_candidates.items():
            if len(matches) == 1:
                target_full = matches[0]
                cnt_skel = val_counts.get(skel, 0)
                cnt_full = val_counts.get(target_full, 0)
                ctx_score = evaluate_context_correlation(df, col, skel, target_full)
                conf = round(min(0.88, 0.80 + (0.04 if cnt_full >= cnt_skel else 0.0) + (ctx_score * 0.04)), 2)
                proposals.append({
                    "column": col,
                    "original_value": skel,
                    "normalized_value": target_full,
                    "confidence": conf,
                    "method": "consonant_skeleton_abbreviation",
                    "issue_type": IssueTaxonomy.SEMANTIC.value,
                    "reason": f"Candidate abbreviation '{skel}' matches consonant skeleton/subsequence of '{target_full}'",
                    "affected_rows": cnt_skel,
                    "is_safe": conf >= 0.85,
                    "is_ambiguous": False,
                })
            else:
                proposals.append({
                    "column": col,
                    "original_value": skel,
                    "normalized_value": matches[0],
                    "confidence": 0.68,
                    "method": "ambiguous_consonant_abbreviation",
                    "issue_type": IssueTaxonomy.SEMANTIC.value,
                    "reason": f"Candidate abbreviation '{skel}' matches multiple terms: {', '.join(matches)}",
                    "affected_rows": val_counts.get(skel, 0),
                    "is_safe": False,
                    "is_ambiguous": True,
                    "possible_meanings": matches,
                    "candidate_meanings": matches,
                })

        # 5d. Unresolved / Out-of-Vocabulary Abbreviation (e.g. 'UET', 'XYZ', 'UNK_CODE' with no dataset match)
        all_resolved = set(acronym_candidates.keys()) | set(prefix_candidates.keys()) | set(consonant_candidates.keys())
        for short_tok in short_tokens:
            if short_tok in all_resolved:
                continue
            s_clean = clean_unicode_and_whitespace(short_tok)
            if s_clean.isupper() and 2 <= len(s_clean) <= 10:
                # Check if it is a standalone unexpanded abbreviation / code
                proposals.append({
                    "column": col,
                    "original_value": short_tok,
                    "normalized_value": short_tok,
                    "confidence": 0.70,
                    "method": "unresolved_abbreviation",
                    "issue_type": IssueTaxonomy.SEMANTIC.value,
                    "reason": f"Abbreviation or code '{short_tok}' has no matching canonical expansion in dataset — review required",
                    "affected_rows": val_counts.get(short_tok, 0),
                    "is_safe": False,
                    "is_ambiguous": True,
                    "candidate_meanings": [],
                })

        # ── 6. Explicit User Mappings if supplied ──
        if extra_abbreviations:
            for val in unique_vals:
                cleaned_val = clean_unicode_and_whitespace(val).lower()
                for k, v in extra_abbreviations.items():
                    if cleaned_val == k.lower():
                        proposals.append({
                            "column": col,
                            "original_value": val,
                            "normalized_value": v,
                            "confidence": 1.0,
                            "method": "user_explicit_mapping",
                            "issue_type": IssueTaxonomy.SEMANTIC.value,
                            "reason": f"Explicit user mapping '{val}' → '{v}'",
                            "affected_rows": val_counts.get(val, 0),
                            "is_safe": True,
                        })

    return proposals


def standardize_values(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    max_cardinality: int = MAX_CARDINALITY_THRESHOLD,
    extra_abbreviations: Optional[dict[str, str]] = None,
    min_confidence: float = 0.85,
    similarity_threshold: Optional[float] = None,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Standardize inconsistent categorical values across DataFrame columns.

    Args:
        df: Source DataFrame.
        columns: Optional columns to standardize.
        max_cardinality: Max unique values per column.
        extra_abbreviations: User-provided explicit dictionary.
        min_confidence: Minimum confidence threshold to auto-commit.
        similarity_threshold: Optional alias for min_confidence.

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).
    """
    result = df.copy()
    effective_min_conf = similarity_threshold if similarity_threshold is not None else min_confidence

    proposals = get_proposed_standardizations(
        df=result,
        columns=columns,
        max_cardinality=max_cardinality,
        extra_abbreviations=extra_abbreviations,
    )

    # Build per-column mapping dictionaries
    col_mappings: dict[str, dict[str, Any]] = {}
    prop_by_col_orig: dict[tuple[str, str], dict[str, Any]] = {}

    for prop in proposals:
        if prop["confidence"] < effective_min_conf or prop.get("is_ambiguous", False):
            continue
        col = prop["column"]
        orig = prop["original_value"]
        new_val = prop["normalized_value"]

        if col not in col_mappings:
            col_mappings[col] = {}

        col_mappings[col][orig] = new_val
        prop_by_col_orig[(col, orig)] = prop

    # Resolve transitive mapping chains (e.g. tps -> TPS -> Thermal Protection System)
    for col, mappings in col_mappings.items():
        for orig in list(mappings.keys()):
            visited = {orig}
            curr = mappings[orig]
            while curr is not None and curr in mappings and curr not in visited:
                visited.add(curr)
                curr = mappings[curr]
            mappings[orig] = curr

    total_cells_modified = 0
    column_details: dict[str, dict[str, Any]] = {}
    applied_mappings: list[dict[str, Any]] = []

    for col, mappings in col_mappings.items():
        if col not in result.columns:
            continue

        for orig, new_val in mappings.items():
            mask = (result[col] == orig) & result[col].notna()
            count = mask.sum()
            if count > 0:
                if new_val is None:
                    result[col] = result[col].mask(mask, pd.NA)
                else:
                    result[col] = result[col].mask(mask, new_val)

                total_cells_modified += count
                if col not in column_details:
                    column_details[col] = {"cells_modified": 0, "mappings": {}}

                column_details[col]["cells_modified"] += count
                column_details[col]["mappings"][orig] = new_val if new_val is not None else "<NaN>"

                if (col, orig) in prop_by_col_orig:
                    applied_p = dict(prop_by_col_orig[(col, orig)])
                    applied_p["normalized_value"] = new_val
                    applied_mappings.append(applied_p)

    logger.info(
        "Semantic standardization: {} cells modified across {} columns (min_conf={})",
        total_cells_modified,
        len(column_details),
        min_confidence,
    )

    record = RepairRecord(
        operation="standardize_values",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        column=", ".join(column_details.keys()) if column_details else None,
        issue_type=IssueTaxonomy.CATEGORICAL.value,
        original_value=f"{total_cells_modified} inconsistent values",
        new_value="standardized canonical representations",
        method="deterministic_8tier_hierarchy",
        confidence=min_confidence,
        reason=f"Standardized {total_cells_modified} inconsistent cell(s) across {len(column_details)} column(s) using 8-tier normalization hierarchy",
        status="applied",
        risk_level="safe",
        details={
            "repair_type": "categorical_standardization",
            "total_cells_modified": total_cells_modified,
            "columns_standardized": {
                col: info["cells_modified"] for col, info in column_details.items()
            },
            "applied_mappings": applied_mappings,
            "min_confidence": min_confidence,
            "status": "applied",
        },
    )

    return result, record


def detect_inconsistent_columns(
    df: pd.DataFrame,
    max_cardinality: int = MAX_CARDINALITY_THRESHOLD,
) -> list[str]:
    """Detect categorical columns that have inconsistent values with confidence >= 0.85."""
    proposals = get_proposed_standardizations(df, max_cardinality=max_cardinality)
    inconsistent_cols = sorted(list(set(p["column"] for p in proposals if p["confidence"] >= 0.85)))
    return inconsistent_cols


# Alias for backward and test compatibility
standardize_categorical_values = standardize_values

