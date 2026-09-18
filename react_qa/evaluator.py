from __future__ import annotations

import json
from pathlib import Path

from .agent import ReActAgent
from .metrics import exact_match, token_f1


def load_examples(path: str | Path) -> list[dict]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(source.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data["data"]


def evaluate(agent: ReActAgent, examples: list[dict], run_dir: str | Path = "runs") -> dict:
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, example in enumerate(examples):
        result = agent.run(example["question"])
        reference = example["answer"]
        row = {
            "id": example.get("id", str(index)), "question": example["question"],
            "reference": reference, "prediction": result.answer,
            "exact_match": exact_match(result.answer, reference),
            "f1": token_f1(result.answer, reference), "status": result.status,
            "steps": len(result.steps),
        }
        rows.append(row)
        agent.save_trace(result, destination / f"trace_{index:04d}.json")
    summary = {
        "count": len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows) if rows else 0.0,
        "f1": sum(row["f1"] for row in rows) / len(rows) if rows else 0.0,
        "average_steps": sum(row["steps"] for row in rows) / len(rows) if rows else 0.0,
        "results": rows,
    }
    (destination / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
