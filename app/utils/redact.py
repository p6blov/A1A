"""PII redaction (Milestone 4).

`redact()` replaces personally identifiable / sensitive substrings with stable
placeholders *before* a prompt leaves the process for the model provider. It runs
in `redact_node` (the first node of the graph) when `ENABLE_REDACTION=true`, so
the conversation reaches the provider already scrubbed.

Design notes
------------
* **Deterministic, dependency-free.** Detection is regex-based, so there is no
  model to load and behavior is reproducible. Free-form *names* are intentionally
  not covered — reliable name detection needs an NER model (e.g. spaCy); add that
  as a separate, opt-in step rather than guessing with regex.
* **Order matters.** Patterns run from most- to least-specific so a broad pattern
  (e.g. phone) can't swallow digits that belong to a narrower one (e.g. card/SSN).
  Replacements are applied in sequence; placeholders contain no digits, so later
  numeric patterns never re-match an already-redacted span.
* **False positives are preferred over leaks.** When a pattern is ambiguous we
  lean toward redacting. Credit cards are the exception: they are validated with
  the Luhn checksum so ordinary 13–16 digit numbers aren't masked.
"""
from __future__ import annotations

import re

EMAIL = "[EMAIL]"
PHONE = "[PHONE]"
CARD = "[CARD]"
SSN = "[SSN]"
IP = "[IP]"
API_KEY = "[API_KEY]"


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Common provider key shapes + bearer tokens. Kept prefix-anchored to avoid
# masking ordinary long words.
_API_KEY_RE = re.compile(
    r"""(?x)
    \b(?:
        AIza[0-9A-Za-z\-_]{35}            # Google API key
      | sk-[A-Za-z0-9]{20,}               # OpenAI-style secret key
      | gh[pousr]_[A-Za-z0-9]{36,}        # GitHub token
      | AKIA[0-9A-Z]{16}                  # AWS access key id
      | xox[baprs]-[A-Za-z0-9-]{10,}      # Slack token
    )\b
    | \bBearer\s+[A-Za-z0-9._\-]{20,}\b   # Authorization: Bearer <token>
    """
)

_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)

# Matches full and (compressed) IPv6 forms. Not a strict validator — it errs
# toward catching address-shaped tokens.
_IPV6_RE = re.compile(
    r"(?<![:.\w])(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}(?![:.\w])"
    r"|(?<![:.\w])(?:[A-Fa-f0-9]{1,4}:){1,7}:(?:[A-Fa-f0-9]{1,4})?(?![:.\w])"
)

# Candidate card: 13–16 digits, optionally split by single spaces or dashes.
_CARD_RE = re.compile(r"\b\d(?:[ -]?\d){12,15}\b")

_SSN_RE = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")

# US/international phone numbers: optional country code, optional area-code
# parens, 7–10 significant digits with common separators.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
)


def _luhn_ok(digits: str) -> bool:
    """Return True if `digits` (string of 0-9) passes the Luhn checksum."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _redact_cards(text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 16 and _luhn_ok(digits):
            return CARD
        return m.group(0)  # not a valid card — leave untouched

    return _CARD_RE.sub(_sub, text)


def redact(text: str) -> str:
    """Return `text` with PII replaced by placeholders.

    Applied most- to least-specific. The graph only calls this when
    `ENABLE_REDACTION` is true (see `app/graph/nodes.py`).
    """
    text = _EMAIL_RE.sub(EMAIL, text)
    text = _API_KEY_RE.sub(API_KEY, text)
    text = _IPV4_RE.sub(IP, text)
    text = _IPV6_RE.sub(IP, text)
    text = _redact_cards(text)
    text = _SSN_RE.sub(SSN, text)
    text = _PHONE_RE.sub(PHONE, text)
    return text
