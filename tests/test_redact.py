"""Tests for PII redaction (Milestone 4)."""
from __future__ import annotations

from app.utils.redact import redact


def test_email():
    assert redact("reach me at jane.doe+test@example.co.uk please") == (
        "reach me at [EMAIL] please"
    )


def test_phone():
    assert "[PHONE]" in redact("call (555) 123-4567")
    assert "[PHONE]" in redact("call +1 555-123-4567")


def test_credit_card_luhn_valid():
    # 4111 1111 1111 1111 is a well-known Luhn-valid test number.
    assert redact("card 4111 1111 1111 1111") == "card [CARD]"


def test_credit_card_invalid_luhn_is_left_alone():
    # Same length but fails Luhn -> not a card, must not be redacted.
    assert redact("order 1234 5678 9012 3456") == "order 1234 5678 9012 3456"


def test_ssn():
    assert redact("ssn 123-45-6789") == "ssn [SSN]"


def test_ipv4():
    assert redact("server at 192.168.1.100 down") == "server at [IP] down"


def test_ipv6():
    assert "[IP]" in redact("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334")


def test_api_keys():
    assert "[API_KEY]" in redact("key AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe")
    assert "[API_KEY]" in redact("Authorization: Bearer abcdef1234567890ghijklmnop")


def test_plain_text_untouched():
    text = "What is the capital of France? It has 67 million people."
    assert redact(text) == text


def test_multiple_in_one_prompt():
    out = redact("email a@b.com phone 555-123-4567 ssn 123-45-6789")
    assert out == "email [EMAIL] phone [PHONE] ssn [SSN]"
