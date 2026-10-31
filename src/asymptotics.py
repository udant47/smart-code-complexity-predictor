"""A small algebra of asymptotic growth terms for Big-O estimation.

Used by ``src.algorithm_analyzer``. Kept separate so the arithmetic of
Big-O (adding sequential costs, multiplying nested ones, dropping
dominated terms) can be tested on its own.

* A ``Mono`` is one product term such as ``n^2``, ``n log n``, ``n·m``,
  ``√n`` or ``2^n``, over one or more *size symbols*. A symbol stands for
  one input size, e.g. ``len(arr)`` or the value of parameter ``n``.
* A ``Cost`` is a sum of terms. Sequential code adds costs (``+``); loop
  bodies are multiplied by the iteration count (``*``). Dominated terms
  are dropped (``n^2 + n -> n^2``), but terms over *different* symbols are
  kept (``n + m`` stays ``n + m``), because neither dominates the other.
* ``Cost.unknown()`` marks a cost that could not be determined. It absorbs
  everything it is combined with, so one unknown part makes the whole
  estimate unknown instead of silently dropping it.

Everything is a worst-case upper bound up to constant factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional

Power = Fraction


@dataclass(frozen=True)
class Mono:
    """Product over symbols of ``s^power · log^logs s · growth(s)``.

    ``powers``, ``logs``: sorted tuples of (symbol, exponent).
    ``growth``: sorted tuple of (symbol, kind, base); kind "exp" means
    ``base^s`` and kind "fact" means ``s!``.
    """

    powers: tuple[tuple[str, Power], ...] = ()
    logs: tuple[tuple[str, int], ...] = ()
    growth: tuple[tuple[str, str, int], ...] = ()

    # Constructors ------------------------------------------------------ #
    @staticmethod
    def const() -> "Mono":
        return Mono()

    @staticmethod
    def sym(symbol: str, power: Power = Fraction(1)) -> "Mono":
        return Mono(powers=((symbol, Fraction(power)),))

    @staticmethod
    def log(symbol: str, k: int = 1) -> "Mono":
        return Mono(logs=((symbol, k),))

    @staticmethod
    def exp(symbol: str, base: int) -> "Mono":
        return Mono(growth=((symbol, "exp", base),))

    @staticmethod
    def factorial(symbol: str) -> "Mono":
        return Mono(growth=((symbol, "fact", 0),))

    # Queries ----------------------------------------------------------- #
    @property
    def is_const(self) -> bool:
        return not self.powers and not self.logs and not self.growth

    def symbols(self) -> set[str]:
        return {s for s, _ in self.powers} | {s for s, _ in self.logs} | {s for s, _, _ in self.growth}

    def power_of(self, symbol: str) -> Power:
        return dict(self.powers).get(symbol, Fraction(0))

    def log_of(self, symbol: str) -> int:
        return dict(self.logs).get(symbol, 0)

    def without(self, symbol: str) -> "Mono":
        return Mono(
            tuple(p for p in self.powers if p[0] != symbol),
            tuple(p for p in self.logs if p[0] != symbol),
            tuple(g for g in self.growth if g[0] != symbol),
        )

    def _key(self, symbol: str) -> tuple:
        g = {s: (k, b) for s, k, b in self.growth}.get(symbol)
        fact = 1 if g and g[0] == "fact" else 0
        base = g[1] if g and g[0] == "exp" else 0
        return (fact, base, self.power_of(symbol), self.log_of(symbol))

    def dominated_by(self, other: "Mono") -> bool:
        """True if self grows no faster than other in every symbol."""
        return all(self._key(s) <= other._key(s) for s in self.symbols() | other.symbols())

    # Arithmetic -------------------------------------------------------- #
    def __mul__(self, other: "Mono") -> "Mono":
        powers = dict(self.powers)
        for s, p in other.powers:
            powers[s] = powers.get(s, Fraction(0)) + p
        logs = dict(self.logs)
        for s, k in other.logs:
            logs[s] = logs.get(s, 0) + k
        growth: dict[str, tuple[str, int]] = {s: (k, b) for s, k, b in self.growth}
        for s, k, b in other.growth:
            if s in growth:
                k0, b0 = growth[s]
                if k0 == k == "exp":
                    growth[s] = ("exp", b0 * b)
                elif "fact" in (k0, k):
                    growth[s] = ("fact", 0)
            else:
                growth[s] = (k, b)
        return Mono(
            tuple(sorted((s, p) for s, p in powers.items() if p != 0)),
            tuple(sorted((s, k) for s, k in logs.items() if k != 0)),
            tuple(sorted((s, k, b) for s, (k, b) in growth.items())),
        )

    def rename(self, mapping: dict[str, str]) -> "Mono":
        """Rename symbols (merging those mapped to the same name)."""
        out = Mono()
        for s, p in self.powers:
            out = out * Mono.sym(mapping.get(s, s), p)
        for s, k in self.logs:
            out = out * Mono.log(mapping.get(s, s), k)
        for s, kind, b in self.growth:
            t = mapping.get(s, s)
            out = out * (Mono.exp(t, b) if kind == "exp" else Mono.factorial(t))
        return out

    def substitute(self, mapping: dict[str, "Mono"]) -> "Mono":
        """Replace each symbol by a size term (for calls between functions).

        Only polynomial powers can be substituted by arbitrary terms; a log
        or growth factor of a substituted symbol requires the replacement
        to be a plain power of a single symbol (otherwise ValueError).
        """
        out = Mono()
        for s, p in self.powers:
            if s not in mapping:
                out = out * Mono.sym(s, p)
                continue
            rep = mapping[s]
            if rep.is_const:
                continue
            if p.denominator == 1:
                for _ in range(p.numerator):
                    out = out * rep
            elif _single_symbol(rep) is not None or (len(rep.powers) == 1 and not rep.logs and not rep.growth):
                out = out * _power(rep, p)
            else:
                raise ValueError("fractional power of a composite size")
        for s, k in self.logs:
            target = _single_symbol(mapping[s]) if s in mapping else s
            if target is None:
                if s in mapping and mapping[s].is_const:
                    continue
                raise ValueError("log of a composite size")
            out = out * Mono.log(target, k)
        for s, kind, b in self.growth:
            target = _single_symbol(mapping[s]) if s in mapping else s
            if target is None:
                if s in mapping and mapping[s].is_const:
                    continue
                raise ValueError("exponential of a composite size")
            out = out * (Mono.exp(target, b) if kind == "exp" else Mono.factorial(target))
        return out

    # Rendering --------------------------------------------------------- #
    def render(self, names: Optional[dict[str, str]] = None) -> str:
        names = names or {}
        order = sorted(self.symbols(), key=lambda s: list(names).index(s) if s in names else 99)
        parts = []
        for s in order:
            n = names.get(s, s)
            piece = []
            p = self.power_of(s)
            if p:
                piece.append(_render_power(n, p))
            k = self.log_of(s)
            if k:
                piece.append(f"log {n}" if k == 1 else f"log^{k} {n}")
            text = " ".join(piece)
            for gs, kind, b in self.growth:
                if gs == s:
                    g = f"{n}!" if kind == "fact" else f"{b}^{n}"
                    text = f"{text}·{g}" if text else g
            parts.append(text)
        return "·".join(parts) if parts else "1"

    def degree(self) -> tuple:
        """Sort key: higher means faster growing (for display order only)."""
        return (
            any(k == "fact" for _, k, _ in self.growth),
            max((b for _, k, b in self.growth if k == "exp"), default=0),
            sum(p for _, p in self.powers),
            sum(k for _, k in self.logs),
        )


def _single_symbol(m: Mono) -> Optional[str]:
    if len(m.powers) == 1 and not m.logs and not m.growth:
        return m.powers[0][0]
    return None


def _power(m: Mono, p: Fraction) -> Mono:
    (s, q), = m.powers
    return Mono.sym(s, q * p)


def _render_power(name: str, p: Fraction) -> str:
    if p == 1:
        return name
    if p == Fraction(1, 2):
        return f"√{name}"
    if p.denominator == 1:
        return f"{name}^{p.numerator}"
    return f"{name}^{float(p):.3g}"


@dataclass(frozen=True)
class Cost:
    """A sum of Mono terms, or unknown."""

    terms: frozenset[Mono] = frozenset()
    is_unknown: bool = False

    @staticmethod
    def zero() -> "Cost":
        return Cost()

    @staticmethod
    def const() -> "Cost":
        return Cost(frozenset({Mono.const()}))

    @staticmethod
    def of(*monos: Mono) -> "Cost":
        return Cost(frozenset(monos)).simplified()

    @staticmethod
    def unknown() -> "Cost":
        return Cost(frozenset(), True)

    @staticmethod
    def total(costs: Iterable["Cost"]) -> "Cost":
        out = Cost.zero()
        for c in costs:
            out = out + c
        return out

    def __add__(self, other: "Cost") -> "Cost":
        if self.is_unknown or other.is_unknown:
            return Cost.unknown()
        return Cost(self.terms | other.terms).simplified()

    def __mul__(self, other: "Cost") -> "Cost":
        if self.is_unknown or other.is_unknown:
            return Cost.unknown()
        return Cost(frozenset(a * b for a in self.terms for b in other.terms)).simplified()

    def times(self, m: Mono) -> "Cost":
        return self * Cost.of(m)

    def simplified(self) -> "Cost":
        if self.is_unknown:
            return self
        # Distinct terms can never dominate each other mutually (equal keys
        # in every symbol means equal terms), so this is order-independent.
        kept = [t for t in self.terms if not any(u != t and t.dominated_by(u) for u in self.terms)]
        return Cost(frozenset(kept))

    def symbols(self) -> set[str]:
        return set().union(*(t.symbols() for t in self.terms)) if self.terms else set()

    def map_terms(self, fn) -> "Cost":
        if self.is_unknown:
            return self
        return Cost(frozenset(fn(t) for t in self.terms)).simplified()

    def collapse(self, symbol: str = "n") -> "Cost":
        """Treat every size as the same n (n·m -> n^2, n + m -> n)."""
        return self.map_terms(lambda t: t.rename({s: symbol for s in t.symbols()}))

    def render(self, names: Optional[dict[str, str]] = None) -> str:
        if self.is_unknown:
            return "Unknown"
        if not self.terms:
            return "O(1)"
        rank = {s: i for i, s in enumerate(names or {})}
        by_name = sorted(self.terms, key=lambda t: sorted(rank.get(s, 99) for s in t.symbols()))
        ordered = sorted(by_name, key=lambda t: t.degree(), reverse=True)  # stable
        return "O(" + " + ".join(t.render(names) for t in ordered) + ")"
