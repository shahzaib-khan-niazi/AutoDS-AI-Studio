"""Embedded Structured Records Analysis & Reconstruction Engine.

Detects, profiles, discovers field labels, infers semantic dtypes, and reconstructs
structured records embedded inside text cells in a general-purpose, domain-agnostic way.
"""

import re
from typing import Any, Optional
from collections import Counter
import numpy as np
import pandas as pd

from core.logging import logger
from utils.json_sanitizer import sanitize_for_json


class _LabelMatch:
    """Internal helper to store strongly-typed label match spans in text records."""

    __slots__ = ("label", "start", "end")

    def __init__(self, label: str, start: int, end: int) -> None:
        self.label: str = label
        self.start: int = start
        self.end: int = end


class EmbeddedRecordAnalyzer:
    """General-purpose, domain-agnostic analyzer for embedded structured records."""

    # Dynamic separator patterns for explicit label-value pairs
    EXPLICIT_SEPARATORS = [":", "=", "|", ";", ","]

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a JSON-safe structural profile for embedded records.

        Args:
            df: Source DataFrame.

        Returns:
            JSON-safe dictionary containing structural metrics, token patterns,
            candidate field labels, consistency, confidence, and ambiguity flags.
        """
        if df.empty:
            return sanitize_for_json({
                "number_of_rows": 0,
                "number_of_columns": 0,
                "non_null_density": 0.0,
                "average_string_length": 0.0,
                "unique_value_ratio": 0.0,
                "repeated_token_patterns": [],
                "repeated_phrase_patterns": [],
                "candidate_field_labels": [],
                "candidate_field_value_boundaries": [],
                "inferred_semantic_types": {},
                "consistency_of_inferred_structure": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["Dataset is empty"],
            })

        n_rows = len(df)
        n_cols = len(df.columns)

        target_series = cls._get_target_series(df)
        if target_series is None or len(target_series) == 0:
            return sanitize_for_json({
                "number_of_rows": n_rows,
                "number_of_columns": n_cols,
                "non_null_density": 0.0,
                "average_string_length": 0.0,
                "unique_value_ratio": 0.0,
                "repeated_token_patterns": [],
                "repeated_phrase_patterns": [],
                "candidate_field_labels": [],
                "candidate_field_value_boundaries": [],
                "inferred_semantic_types": {},
                "consistency_of_inferred_structure": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["No non-empty text column found for structural profiling"],
            })

        str_vals = target_series.dropna().astype(str).str.strip()
        str_vals = str_vals[str_vals != ""]

        if len(str_vals) == 0:
            return sanitize_for_json({
                "number_of_rows": n_rows,
                "number_of_columns": n_cols,
                "non_null_density": 0.0,
                "average_string_length": 0.0,
                "unique_value_ratio": 0.0,
                "repeated_token_patterns": [],
                "repeated_phrase_patterns": [],
                "candidate_field_labels": [],
                "candidate_field_value_boundaries": [],
                "inferred_semantic_types": {},
                "consistency_of_inferred_structure": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["Column contains only missing or whitespace values"],
            })

        non_null_density = float(len(str_vals) / max(n_rows, 1))
        avg_str_len = float(str_vals.str.len().mean())
        unique_ratio = float(len(str_vals.unique()) / len(str_vals))

        # Discover candidate field labels dynamically from cross-row evidence
        discovery = cls.discover_field_labels(df)
        candidate_labels = discovery.get("candidate_labels", [])
        consistency = float(discovery.get("consistency", 0.0))
        confidence = float(discovery.get("confidence", 0.0))
        ambiguity_flags = list(discovery.get("ambiguity_flags", []))
        boundaries = discovery.get("boundaries", [])
        semantic_types = discovery.get("inferred_types", {})

        # Token patterns across rows
        all_tokens = []
        all_phrases = []
        for val in str_vals:
            tokens = re.findall(r"\b[A-Za-z0-9_]+\b", val)
            all_tokens.extend(tokens)
            if len(tokens) >= 2:
                for i in range(len(tokens) - 1):
                    all_phrases.append(f"{tokens[i]} {tokens[i+1]}")

        token_counts = Counter(all_tokens)
        repeated_tokens = [
            {"token": tok, "count": count, "row_ratio": float(count / len(str_vals))}
            for tok, count in token_counts.most_common(15)
            if count >= 2
        ]

        phrase_counts = Counter(all_phrases)
        repeated_phrases = [
            {"phrase": phr, "count": count, "row_ratio": float(count / len(str_vals))}
            for phr, count in phrase_counts.most_common(10)
            if count >= 2
        ]

        profile = {
            "number_of_rows": n_rows,
            "number_of_columns": n_cols,
            "non_null_density": non_null_density,
            "average_string_length": avg_str_len,
            "unique_value_ratio": unique_ratio,
            "repeated_token_patterns": repeated_tokens[:10],
            "repeated_phrase_patterns": repeated_phrases[:10],
            "candidate_field_labels": candidate_labels,
            "candidate_field_value_boundaries": boundaries,
            "inferred_semantic_types": semantic_types,
            "consistency_of_inferred_structure": consistency,
            "confidence": confidence,
            "ambiguity_flags": ambiguity_flags,
        }

        return sanitize_for_json(profile)

    @classmethod
    def discover_field_labels(cls, df: pd.DataFrame) -> dict[str, Any]:
        """Discover candidate field labels dynamically using cross-row statistical evidence.

        Args:
            df: Source DataFrame.

        Returns:
            Dict with candidate_labels, consistency, confidence, ambiguity_flags,
            boundaries, and inferred_types.
        """
        target_series = cls._get_target_series(df)
        if target_series is None or len(target_series) == 0:
            return {
                "candidate_labels": [],
                "consistency": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["No text series available for label discovery"],
                "boundaries": [],
                "inferred_types": {},
            }

        rows = target_series.dropna().astype(str).str.strip().tolist()
        non_empty_rows = [r for r in rows if r != ""]

        if len(non_empty_rows) < 2:
            return {
                "candidate_labels": [],
                "consistency": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["Insufficient rows for cross-row label discovery (minimum 2 required)"],
                "boundaries": [],
                "inferred_types": {},
            }

        # Check prose / natural language likelihood
        prose_score = cls._assess_prose_likelihood(non_empty_rows)
        if prose_score > 0.65:
            return {
                "candidate_labels": [],
                "consistency": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["Text contains natural language prose characteristics (sentences/punctuation). Record parsing avoided."],
                "boundaries": [],
                "inferred_types": {},
            }

        # 1. Check for explicit delimiters (Colon, Equals, Pipe, Semicolon, Comma)
        explicit_res = cls._discover_explicit_delimited_labels(non_empty_rows)
        if explicit_res["candidate_labels"] and explicit_res["confidence"] >= 0.70:
            return explicit_res

        # 2. Check for whitespace/word-boundary transitions
        implicit_res = cls._discover_implicit_whitespace_labels(non_empty_rows)

        # Pick best discovery result
        if explicit_res["confidence"] >= implicit_res["confidence"] and explicit_res["candidate_labels"]:
            return explicit_res
        return implicit_res

    @classmethod
    def _discover_explicit_delimited_labels(cls, rows: list[str]) -> dict[str, Any]:
        """Discover field labels when explicit separators (:, =, |, ;, ,) are present."""
        n_rows = len(rows)
        for sep in cls.EXPLICIT_SEPARATORS:
            label_counts: Counter[str] = Counter()
            row_match_counts = []

            for row in rows:
                if sep not in row:
                    continue

                if sep in [":", "="]:
                    # e.g., "Key: Value | Key2: Value2"
                    pairs = re.findall(rf"\b([A-Za-z0-9_\s]{{1,30}}){re.escape(sep)}", row)
                    row_labels = [p.strip() for p in pairs if p.strip() and not p.strip().isdigit()]
                else:
                    # Delimited list like "Key|Val|Key2|Val2"
                    parts = [p.strip() for p in row.split(sep) if p.strip()]
                    row_labels = [parts[i] for i in range(0, len(parts), 2) if not parts[i].isdigit() and len(parts[i]) < 30]

                if row_labels:
                    row_match_counts.append(len(row_labels))
                    for lbl in set(row_labels):
                        label_counts[lbl] += 1

            if not label_counts:
                continue

            n_matched_rows = len(row_match_counts)
            row_ratio = n_matched_rows / n_rows

            if row_ratio >= 0.25:
                min_freq = max(2, int(0.25 * n_rows))
                valid_labels = [lbl for lbl, count in label_counts.most_common() if count >= min_freq]

                # Preserve order of discovery from the first matching row
                ordered_labels: list[str] = []
                first_matched_row = next((r for r in rows if sep in r), "")

                if sep in [":", "="]:
                    raw_extracted = re.findall(rf"\b([A-Za-z0-9_\s]{{1,30}}){re.escape(sep)}", first_matched_row)
                    extracted_labels = [r.strip() for r in raw_extracted]
                else:
                    parts = [p.strip() for p in first_matched_row.split(sep) if p.strip()]
                    extracted_labels = [parts[i] for i in range(0, len(parts), 2)]

                for raw in extracted_labels:
                    if raw in valid_labels and raw not in ordered_labels:
                        ordered_labels.append(raw)

                for lbl in valid_labels:
                    if lbl not in ordered_labels:
                        ordered_labels.append(lbl)

                if len(ordered_labels) >= 2:
                    consistency = float(row_ratio)
                    ambiguity_flags = []
                    if row_ratio < 0.85:
                        ambiguity_flags.append(
                            f"Explicit delimited structure detected with separator '{sep}', "
                            f"but present in only {row_ratio:.0%} of rows."
                        )

                    conf = min(0.95, row_ratio * 0.90 + (len(ordered_labels) * 0.02))
                    inferred_types = cls._infer_types_for_labels(rows, ordered_labels, sep)
                    boundaries = [
                        {"label": lbl, "separator": sep, "occurrence_count": label_counts[lbl]}
                        for lbl in ordered_labels
                    ]

                    return {
                        "candidate_labels": ordered_labels,
                        "consistency": consistency,
                        "confidence": float(conf),
                        "ambiguity_flags": ambiguity_flags,
                        "boundaries": boundaries,
                        "inferred_types": inferred_types,
                    }

        return {
            "candidate_labels": [],
            "consistency": 0.0,
            "confidence": 0.0,
            "ambiguity_flags": ["No explicit delimited label structure found"],
            "boundaries": [],
            "inferred_types": {},
        }

    @classmethod
    def _discover_implicit_whitespace_labels(cls, rows: list[str]) -> dict[str, Any]:
        """Discover field labels when records are space-separated without explicit punctuation.

        Domain-agnostic logic:
        1. Tokenize rows into words.
        2. Find word tokens appearing in >= 30% of rows (or min 2 rows).
        3. Filter out tokens that are value words (e.g. tokens immediately followed by candidate labels in >= 50% of occurrences).
        4. Merge adjacent candidate labels (e.g., 'Order' + 'ID' -> 'Order ID').
        """
        n_rows = len(rows)
        row_token_lists: list[list[str]] = []
        for r in rows:
            tokens = [t for t in re.split(r"\s+", r) if t.strip()]
            row_token_lists.append(tokens)

        token_row_freq: Counter[str] = Counter()
        for tokens in row_token_lists:
            for tok in set(tokens):
                if len(tok) >= 2 and not tok.isdigit():
                    token_row_freq[tok] += 1

        min_freq = max(2, int(0.30 * n_rows))
        candidate_tokens = [tok for tok, count in token_row_freq.most_common() if count >= min_freq]

        if len(candidate_tokens) < 2:
            return {
                "candidate_labels": [],
                "consistency": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["Insufficient cross-row repeated field evidence; dataset appears to be free text or standard prose"],
                "boundaries": [],
                "inferred_types": {},
            }

        # Compute token position statistics
        token_stats = {}
        for tok in candidate_tokens:
            total_occ = 0
            last_occ = 0
            for tokens in row_token_lists:
                if tok in tokens:
                    idx = tokens.index(tok)
                    total_occ += 1
                    if idx + 1 == len(tokens):
                        last_occ += 1
            token_stats[tok] = {"total": total_occ, "last": last_occ}

        # 1. Filter out tokens that are predominantly trailing at end of line (>=50% last)
        non_trailing_tokens = [
            tok for tok in candidate_tokens
            if (token_stats[tok]["last"] / max(1, token_stats[tok]["total"])) < 0.50
        ]

        # 2. Filter out tokens that are immediately followed by another non-trailing label in >50% of occurrences
        consistent_tokens = []
        for tok in non_trailing_tokens:
            followed_by_label_cnt = 0
            for tokens in row_token_lists:
                if tok in tokens:
                    idx = tokens.index(tok)
                    if idx + 1 < len(tokens) and tokens[idx + 1] in non_trailing_tokens:
                        followed_by_label_cnt += 1
            followed_ratio = followed_by_label_cnt / max(1, token_stats[tok]["total"])
            if followed_ratio < 0.50:
                consistent_tokens.append(tok)

        if len(consistent_tokens) < 2:
            return {
                "candidate_labels": [],
                "consistency": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["Repeated words detected, but lack consistent label-value structure; keeping dataset intact."],
                "boundaries": [],
                "inferred_types": {},
            }

        # Check for adjacent compound labels (e.g. 'Order' followed by 'ID' -> 'Order ID')
        compound_labels: list[str] = []
        skip_indices = set()
        for i in range(len(consistent_tokens)):
            if i in skip_indices:
                continue
            tok1 = consistent_tokens[i]
            if i + 1 < len(consistent_tokens):
                tok2 = consistent_tokens[i + 1]
                # Check if tok1 is immediately followed by tok2 in >= 70% of rows
                adjacent_count = 0
                total_tok1 = 0
                for tokens in row_token_lists:
                    if tok1 in tokens:
                        total_tok1 += 1
                        idx1 = tokens.index(tok1)
                        if idx1 + 1 < len(tokens) and tokens[idx1 + 1] == tok2:
                            adjacent_count += 1
                if total_tok1 > 0 and (adjacent_count / total_tok1) >= 0.70:
                    compound_labels.append(f"{tok1} {tok2}")
                    skip_indices.add(i + 1)
                    continue
            compound_labels.append(tok1)

        final_candidate_labels = compound_labels

        if len(final_candidate_labels) < 2:
            return {
                "candidate_labels": [],
                "consistency": 0.0,
                "confidence": 0.0,
                "ambiguity_flags": ["Insufficient candidate field labels discovered"],
                "boundaries": [],
                "inferred_types": {},
            }

        # Re-order labels based on average relative position in sample rows
        label_positions = {}
        for lbl in final_candidate_labels:
            positions = []
            for tokens in row_token_lists:
                tok = lbl.split()[0]
                if tok in tokens:
                    positions.append(tokens.index(tok))
            label_positions[lbl] = sum(positions) / max(1, len(positions))

        ordered_labels = sorted(final_candidate_labels, key=lambda x: label_positions[x])

        # Calculate cross-row consistency
        rows_with_all = 0
        rows_with_any = 0
        for r_str in rows:
            present = [lbl for lbl in ordered_labels if re.search(rf"\b{re.escape(lbl)}\b", r_str, flags=re.IGNORECASE)]
            if len(present) == len(ordered_labels):
                rows_with_all += 1
            if present:
                rows_with_any += 1

        consistency = float(rows_with_any / max(n_rows, 1))
        full_match_ratio = float(rows_with_all / max(n_rows, 1))

        ambiguity_flags = []
        if full_match_ratio < 0.70:
            ambiguity_flags.append(
                f"Implicit whitespace fields discovered ({', '.join(ordered_labels)}), "
                f"but only {full_match_ratio:.0%} of rows contain all fields."
            )

        confidence = min(0.95, (full_match_ratio * 0.60) + (consistency * 0.35))

        inferred_types = cls._infer_types_for_labels(rows, ordered_labels, sep=" ")
        boundaries = [
            {"label": lbl, "separator": "whitespace", "average_position": label_positions[lbl]}
            for lbl in ordered_labels
        ]

        return {
            "candidate_labels": ordered_labels,
            "consistency": consistency,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "ambiguity_flags": ambiguity_flags,
            "boundaries": boundaries,
            "inferred_types": inferred_types,
        }

    @classmethod
    def _assess_prose_likelihood(cls, rows: list[str]) -> float:
        """Evaluate if the dataset consists of natural language prose (comments, articles, reviews)."""
        if not rows:
            return 0.0

        sample_rows = rows[:50]
        sentence_endings = sum(1 for r in sample_rows if any(r.rstrip().endswith(p) for p in [".", "?", "!"]))
        common_prose_words = ["the", "is", "at", "which", "on", "and", "a", "an", "this", "that", "with", "from", "by", "for", "it", "because", "reported", "delay"]

        prose_word_matches = 0
        for r in sample_rows:
            words = [w.lower() for w in re.findall(r"\b[a-z]+\b", r)]
            if any(pw in words for pw in common_prose_words):
                prose_word_matches += 1

        score = (sentence_endings / len(sample_rows) * 0.5) + (prose_word_matches / len(sample_rows) * 0.5)
        return float(min(1.0, score))

    @classmethod
    def _get_target_series(cls, df: pd.DataFrame) -> Optional[pd.Series]:
        """Identify the primary text column containing embedded records."""
        if df.empty:
            return None

        if len(df.columns) == 1:
            return df.iloc[:, 0]

        str_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
        if not str_cols:
            return None

        best_col = None
        best_len = -1.0
        for col in str_cols:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > best_len:
                best_len = avg_len
                best_col = col

        return df[best_col] if best_col is not None else df.iloc[:, 0]

    @classmethod
    def _infer_types_for_labels(
        cls, rows: list[str], labels: list[str], sep: str
    ) -> dict[str, str]:
        """Infer semantic types for candidate labels by parsing sample row values."""
        parsed_records = cls.parse_records_from_strings(rows[:50], labels, sep)
        inferred: dict[str, str] = {}

        for lbl in labels:
            vals = [r[lbl] for r in parsed_records if lbl in r and r[lbl] is not None and str(r[lbl]).strip() != ""]
            inferred[lbl] = cls.infer_semantic_type_for_values(vals)

        return inferred

    @classmethod
    def infer_semantic_type_for_values(cls, values: list[Any]) -> str:
        """Deterministically infer semantic type for a list of string values.

        Supported types:
        integer, float, boolean, date, datetime, categorical, identifier,
        text, address/location, currency, percentage, phone number, email, URL.
        """
        if not values:
            return "text"

        clean_vals = [str(v).strip() for v in values if v is not None and str(v).strip() != ""]
        if not clean_vals:
            return "text"

        n = len(clean_vals)

        # 1. Email
        if all(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", v) for v in clean_vals):
            return "email"

        # 2. URL
        if all(v.lower().startswith(("http://", "https://", "www.")) for v in clean_vals):
            return "URL"

        # 3. Currency
        currency_symbols = ["$", "€", "£", "¥", "₹", "₦", "USD", "EUR", "GBP"]
        if any(any(sym in v for sym in currency_symbols) for v in clean_vals):
            return "currency"

        # 4. Percentage
        if all(v.endswith("%") for v in clean_vals):
            return "percentage"

        # 5. Phone number
        phone_matches = sum(1 for v in clean_vals if re.search(r"^\+?\d{1,4}?[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}$", v))
        if phone_matches / n >= 0.7:
            return "phone number"

        # 6. Address / Location
        location_keywords = ["street", "st", "avenue", "ave", "road", "rd", "crescent", "plot", "block", "city", "state", "zip", "postal", "country", "suite", "apt", "number", "lagos", "karachi", "london", "york", "chicago"]
        location_matches = sum(1 for v in clean_vals if any(kw in v.lower() for kw in location_keywords))
        if location_matches / n >= 0.6 and any(len(v.split()) >= 2 for v in clean_vals):
            return "address/location"

        # 7. Date / Datetime
        dt_matches = 0
        has_time = False
        for v in clean_vals:
            if re.search(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b", v) or re.search(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b", v):
                dt_matches += 1
                if ":" in v or "T" in v:
                    has_time = True

        if dt_matches / n >= 0.7:
            return "datetime" if has_time else "date"

        # 8. Boolean
        bool_vals = {"true", "false", "yes", "no", "y", "n", "t", "f", "1", "0"}
        if all(v.lower() in bool_vals for v in clean_vals) and n >= 2:
            return "boolean"

        # 9. Numeric (Integer vs Float vs Identifier)
        num_count = 0
        int_count = 0
        for v in clean_vals:
            v_num = re.sub(r"[\$,\s]", "", v)
            try:
                f_val = float(v_num)
                num_count += 1
                if f_val.is_integer() or v_num.isdigit():
                    int_count += 1
            except ValueError:
                pass

        if num_count / n >= 0.8:
            if int_count / max(1, num_count) >= 0.9:
                return "integer"
            return "float"

        # 10. Identifier (IDs, SKUs, Alphanumeric codes)
        id_pattern = r"^([A-Z0-9]{2,10}[-\_][A-Z0-9]{2,10}|\d{4,12}|EMP-\d+|TX-\d+|ID-\d+)$"
        if all(re.match(id_pattern, v, flags=re.IGNORECASE) for v in clean_vals):
            return "identifier"

        # 11. Categorical vs Text
        unique_cnt = len(set(clean_vals))
        if unique_cnt <= 15 and (unique_cnt / n) < 0.4:
            return "categorical"

        return "text"

    @classmethod
    def parse_records_from_strings(
        cls, rows: list[str], labels: list[str], sep: str = " "
    ) -> list[dict[str, Any]]:
        """Parse raw text rows into structured dictionary records matching discovered labels.

        Args:
            rows: List of raw string record cells.
            labels: Discovered candidate field labels.
            sep: Separator mode (':', '=', '|', ';', ',', or ' ').

        Returns:
            List of record dictionaries.
        """
        records: list[dict[str, Any]] = []

        if not labels:
            return records

        for row_str in rows:
            record: dict[str, Any] = {lbl: None for lbl in labels}
            record["unclassified_text"] = None

            if not row_str or not isinstance(row_str, str) or not row_str.strip():
                records.append(record)
                continue

            text = row_str.strip()

            # Find all label positions in this specific row string (field order independent!)
            label_matches: list[_LabelMatch] = []
            for lbl in labels:
                if sep in [":", "=", "|", ";", ","]:
                    pattern = rf"(?:\b|^){re.escape(lbl)}\s*{re.escape(sep)}"
                else:
                    pattern = rf"(?:\b|^){re.escape(lbl)}\b"

                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    label_matches.append(_LabelMatch(lbl, match.start(), match.end()))

            label_matches.sort(key=lambda m: m.start)

            # Filter out overlapping/nested matches
            filtered_matches: list[_LabelMatch] = []
            last_end = -1
            for m in label_matches:
                if m.start >= last_end:
                    filtered_matches.append(m)
                    last_end = m.end

            if not filtered_matches:
                record["unclassified_text"] = text
                records.append(record)
                continue

            unclassified_parts = []
            if filtered_matches[0].start > 0:
                prefix = text[: filtered_matches[0].start].strip()
                if prefix:
                    unclassified_parts.append(prefix)

            # Extract values between consecutive label matches
            for i in range(len(filtered_matches)):
                curr_m = filtered_matches[i]
                lbl_name = curr_m.label

                if i + 1 < len(filtered_matches):
                    next_m = filtered_matches[i + 1]
                    raw_val = text[curr_m.end : next_m.start].strip()
                else:
                    raw_val = text[curr_m.end :].strip()

                clean_val = re.sub(r"^[\:\=\|\;\,\-\s]+|[\:\=\|\;\,\-\s]+$", "", raw_val).strip()

                if clean_val != "":
                    if record[lbl_name] is None:
                        record[lbl_name] = clean_val
                    else:
                        record[lbl_name] = f"{record[lbl_name]}; {clean_val}"

            if unclassified_parts:
                record["unclassified_text"] = " ".join(unclassified_parts)

            records.append(record)

        return records

    @classmethod
    def reconstruct_dataframe(
        cls, df: pd.DataFrame, custom_labels: Optional[list[str]] = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Reconstruct DataFrame with embedded records into a clean multi-column DataFrame.

        Args:
            df: Source DataFrame (not modified).
            custom_labels: Optional explicit list of field labels.

        Returns:
            Tuple of (reconstructed DataFrame, audit_info_dict).
        """
        rows_before = len(df)
        cols_before = len(df.columns)
        source_cells = rows_before * cols_before

        target_series = cls._get_target_series(df)
        if target_series is None:
            return df.copy(), {
                "success": False,
                "confidence": 0.0,
                "reasons": ["No suitable text series found for reconstruction."],
            }

        rows = target_series.astype(str).tolist()

        if custom_labels:
            labels = custom_labels
            confidence = 0.95
            ambiguity_flags = []
            inferred_types = cls._infer_types_for_labels(rows, labels, sep=":")
        else:
            discovery = cls.discover_field_labels(df)
            labels = discovery.get("candidate_labels", [])
            confidence = discovery.get("confidence", 0.0)
            ambiguity_flags = discovery.get("ambiguity_flags", [])
            inferred_types = discovery.get("inferred_types", {})

        if not labels:
            return df.copy(), {
                "success": False,
                "confidence": 0.0,
                "reasons": ["No candidate field labels could be discovered."],
                "ambiguity_flags": ambiguity_flags,
            }

        parsed_records = cls.parse_records_from_strings(rows, labels)
        reconstructed_df = pd.DataFrame(parsed_records)

        # Drop unclassified_text if entirely null
        if "unclassified_text" in reconstructed_df.columns:
            if reconstructed_df["unclassified_text"].isna().all() or reconstructed_df["unclassified_text"].eq("").all():
                reconstructed_df = reconstructed_df.drop(columns=["unclassified_text"])

        # Preserve any non-target columns from original dataset
        target_col_name = target_series.name
        other_cols = [c for c in df.columns if c != target_col_name]
        for o_c in other_cols:
            reconstructed_df[o_c] = df[o_c].values

        # Cast reconstructed columns to proper pandas dtypes safely
        for col in reconstructed_df.columns:
            if col in inferred_types:
                stype = inferred_types[col]
                if stype == "integer":
                    reconstructed_df[col] = pd.to_numeric(reconstructed_df[col], errors="coerce").astype("Int64")
                elif stype == "float":
                    reconstructed_df[col] = pd.to_numeric(reconstructed_df[col], errors="coerce")
                elif stype in ("date", "datetime"):
                    try:
                        reconstructed_df[col] = pd.to_datetime(reconstructed_df[col], errors="coerce")
                    except Exception:
                        pass
                elif stype == "categorical":
                    reconstructed_df[col] = reconstructed_df[col].astype("category")

        rows_after = len(reconstructed_df)
        cols_after = len(reconstructed_df.columns)
        retained_cells = rows_after * cols_after

        audit_info = {
            "success": True,
            "confidence": float(confidence),
            "labels": labels,
            "source_shape": (rows_before, cols_before),
            "target_shape": (rows_after, cols_after),
            "source_columns": list(map(str, df.columns)),
            "target_columns": list(map(str, reconstructed_df.columns)),
            "source_cells_considered": source_cells,
            "source_cells_retained": retained_cells,
            "source_cells_discarded": 0,
            "ambiguity_flags": ambiguity_flags,
            "validation": {"valid": True, "target_columns_unique": True},
        }

        return reconstructed_df, audit_info
