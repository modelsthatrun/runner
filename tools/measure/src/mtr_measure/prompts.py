"""The shared prompt sets for language-model tasks, kept with the tool as part of the protocol.

Each set is prompts/<task>.toml: a version, the generation settings every checkpoint answers with, and
the prompts, each with an id, a category, the prompt and the check a correct answer must pass. The output
step for language models reads a set through load(), and records its version and sha256 next to the answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import MeasureError
from .record import sha256_file

DIR = Path(__file__).with_name("prompts")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    prompt: str
    check: str


@dataclass(frozen=True)
class PromptSet:
    task: str
    version: int
    max_tokens: int
    temperature: float
    thinking: str
    system: str
    prompts: tuple[Prompt, ...]
    sha256: str


def load(task: str) -> PromptSet:
    path = DIR / f"{task}.toml"
    if not path.is_file():
        raise MeasureError("USAGE", f'No prompt set for task "{task}" ({path})')
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        raise MeasureError("USAGE", "Reading prompt sets needs Python 3.11 or newer") from None
    data = tomllib.loads(path.read_text())
    gen = data.get("generation", {})
    prompts = tuple(Prompt(**{k: str(p.get(k, "")).strip() for k in ("id", "category", "prompt", "check")}) for p in data.get("prompts", []))
    problems = []
    if not isinstance(data.get("version"), int) or data["version"] < 1:
        problems.append("version must be a positive integer")
    if not prompts:
        problems.append("no prompts")
    seen: set[str] = set()
    for p in prompts:
        if not SLUG.match(p.id):
            problems.append(f"{p.id!r} is not a lowercase-hyphen id")
        if p.id in seen:
            problems.append(f"{p.id} appears twice")
        seen.add(p.id)
        problems += [f"{p.id}: {k} is empty" for k in ("category", "prompt", "check") if not getattr(p, k)]
    if problems:
        raise MeasureError("USAGE", f"{path.name}: {'; '.join(problems)}")
    return PromptSet(
        task=task,
        version=data["version"],
        max_tokens=int(gen.get("max_tokens", 512)),
        temperature=float(gen.get("temperature", 0)),
        thinking=str(gen.get("thinking", "off")),
        system=str(gen.get("system", "")),
        prompts=prompts,
        sha256=sha256_file(path),
    )
