"""Security helpers shared by backend components."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class HostRule:
    hostname: str
    port: int | None = None


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def default_port_for_scheme(scheme: str) -> int | None:
    lowered = (scheme or "").lower()
    if lowered == "https":
        return 443
    if lowered == "http":
        return 80
    return None


def parse_host_rule(value: str) -> HostRule | None:
    raw = (value or "").strip()
    if not raw:
        return None
    candidate = raw if "://" in raw else f"//{raw}"
    parsed = urlparse(candidate)
    if not parsed.hostname:
        return None
    return HostRule(hostname=parsed.hostname.lower(), port=parsed.port)


def build_allowed_hosts(ls_url: str, extra_hosts: str = "") -> list[HostRule]:
    rules: list[HostRule] = []

    primary = parse_host_rule(ls_url)
    if primary:
        rules.append(primary)

    for token in (extra_hosts or "").split(","):
        rule = parse_host_rule(token)
        if rule and rule not in rules:
            rules.append(rule)

    return rules


def is_allowed_host(parsed_url, allowed_hosts: list[HostRule]) -> bool:
    hostname = (parsed_url.hostname or "").lower()
    if not hostname:
        return False
    resolved_port = parsed_url.port or default_port_for_scheme(parsed_url.scheme)
    for rule in allowed_hosts:
        if hostname != rule.hostname:
            continue
        if rule.port is None or rule.port == resolved_port:
            return True
    return False


def validate_remote_http_url(url: str, allowed_hosts: list[HostRule]):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    if not is_allowed_host(parsed, allowed_hosts):
        raise ValueError(f"Host not allowed for download: {parsed.hostname}")
    return parsed


def same_host(url_a: str, url_b: str) -> bool:
    a = urlparse(url_a)
    b = urlparse(url_b)
    if not a.hostname or not b.hostname:
        return False
    a_port = a.port or default_port_for_scheme(a.scheme)
    b_port = b.port or default_port_for_scheme(b.scheme)
    return a.hostname.lower() == b.hostname.lower() and a_port == b_port


def parse_authorization_token(header_value: str | None) -> str:
    raw = (header_value or "").strip()
    if not raw:
        return ""
    parts = raw.split(None, 1)
    if len(parts) == 2 and parts[0].lower() in {"bearer", "token"}:
        return parts[1].strip()
    return raw


def verify_shared_secret(provided: str | None, expected: str | None) -> bool:
    expected_value = (expected or "").strip()
    if not expected_value:
        return False
    return hmac.compare_digest((provided or "").strip(), expected_value)
