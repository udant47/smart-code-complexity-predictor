"""Detect files that probably contain secrets, without ever reporting the secret itself.

Two layers:
1. File names that conventionally hold credentials (``.env``, ``*.pem``,
   ``id_rsa`` ...). They are never opened.
2. Content patterns in source files: private-key blocks, well-known token
   formats, and assignments of long literals to names like ``password``.

Findings carry only the rule name and line number, never the matched text.
This is a best-effort safeguard, not a secret scanner. A file without
findings may still contain secrets.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

SENSITIVE_NAME_PATTERNS = (
    ".env", ".env.*", "*.env", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.kdbx", "*.asc",
    "id_rsa", "id_rsa.*", "id_dsa", "id_ecdsa", "id_ed25519", ".netrc", "_netrc", ".pypirc", ".npmrc",
    ".git-credentials", "credentials", "credentials.json", "secrets.json", "secrets.yaml", "secrets.yml",
    "*.tfstate", "service-account*.json",
)

CONTENT_RULES: tuple[tuple[str, str], ...] = (
    ("private_key", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"),
    ("aws_access_key_id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("github_token", r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b"),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ("stripe_live_key", r"\b[sr]k_live_[0-9A-Za-z]{20,}\b"),
    ("password_assignment",
     r"(?i)\b(?:password|passwd|pwd|secret|secret_key|api_key|apikey|access_token|auth_token|private_key|"
     r"client_secret)\b\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']"),
)
PLACEHOLDERS = ("changeme", "change_me", "example", "placeholder", "your_", "your-", "xxxx", "dummy", "test",
                "fake", "<", "${", "{{", "os.environ", "getenv", "*****")


@dataclass(frozen=True)
class SecretFinding:
    rule: str
    line: int

    def to_dict(self) -> dict:
        return {"rule": self.rule, "line": self.line}


def is_sensitive_filename(name: str) -> bool:
    lower = name.lower()
    return any(fnmatch.fnmatchcase(lower, pattern) for pattern in SENSITIVE_NAME_PATTERNS)


def scan_text(text: str, max_findings: int = 5) -> list[SecretFinding]:
    """Rule names and line numbers of likely secrets (the values are never returned)."""
    findings: list[SecretFinding] = []
    for name, pattern in CONTENT_RULES:  # plain patterns (re caches them); the security tests forbid compile()
        for m in re.finditer(pattern, text):
            if name == "password_assignment":
                value = m.group(1).lower()
                if any(p in value for p in PLACEHOLDERS):
                    continue
            findings.append(SecretFinding(name, text.count("\n", 0, m.start()) + 1))
            if len(findings) >= max_findings:
                return sorted(findings, key=lambda f: f.line)
    return sorted(findings, key=lambda f: f.line)
