"""Splits a parsed module into analysable units.

Responsibility:
    Walk the AST and collect every function, async function and method
    (including nested functions and methods inside classes) with its
    qualified name, line span, source segment and parent class.

Planned public API (Phase 2):
    extract_functions(tree: ast.Module, source: str) -> list[FunctionUnit]
"""
