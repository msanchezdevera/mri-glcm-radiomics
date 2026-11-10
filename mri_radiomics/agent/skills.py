"""Skills: task-specific instructions the agent loads on demand.

Each skill is a Markdown file with a small front matter block (name, description).
Only the names and descriptions go into the system prompt; the agent pulls the full
body with the `load_skill` tool when a task calls for it. This keeps the base
prompt short while letting behaviour grow by adding files, not code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


def load_skills(directory: Path = DEFAULT_DIR) -> dict[str, Skill]:
    skills = {}
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        _, meta, body = text.split("---", 2)
        info = yaml.safe_load(meta)
        skills[info["name"]] = Skill(info["name"], info["description"].strip(), body.strip())
    return skills


def catalog(skills: dict[str, Skill]) -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
