"""Shared pytest configuration.

Short test IDs
--------------
pytest builds a test's ID from its parameter values and, while the test runs,
stores the ID in the PYTEST_CURRENT_TEST environment variable. Windows limits
an environment variable to 32,767 characters, so a parameter such as
5,000 lines of code makes the test fail with "the environment variable is
longer than 32767 characters". Long string and bytes parameters therefore get
a short ID: the start of the value plus its length.
"""

MAX_ID_CHARS = 40


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, (str, bytes)) and len(val) > MAX_ID_CHARS:
        head = val[:20]
        if isinstance(head, bytes):
            head = head.decode("utf-8", errors="replace")
        head = head.replace("\n", "\\n").replace("\r", "\\r").replace("\0", "\\0")
        kind = "bytes" if isinstance(val, bytes) else "chars"
        return f"{head}...[{len(val)}-{kind}]"
    return None  # pytest's default
