from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .llm import LanguageModel
from .prompts import build_prompt
from .tools import KnowledgeEnvironment

ACTION_RE = re.compile(r"Action\s*:\s*(Search|Lookup|Finish)\[(.*?)\]", re.I | re.S)
THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*?)(?=\n\s*Action\s*:)", re.I | re.S)


@dataclass
class Step:
    index: int
    thought: str
    action: str
    argument: str
    observation: str


@dataclass
class RunResult:
    question: str
    answer: str
    status: str
    steps: list[Step]

    def to_dict(self) -> dict:
        return {**asdict(self), "steps": [asdict(step) for step in self.steps]}


class ReActAgent:
    def __init__(self, model: LanguageModel, environment: KnowledgeEnvironment, max_steps: int = 8) -> None:
        self.model = model
        self.environment = environment
        self.max_steps = max_steps

    @staticmethod
    def parse(output: str) -> tuple[str, str, str]:
        action_match = ACTION_RE.search(output)
        if not action_match:
            raise ValueError("Model output did not contain a valid Action[argument].")
        thought_match = THOUGHT_RE.search(output)
        thought = thought_match.group(1).strip() if thought_match else ""
        return thought, action_match.group(1).title(), action_match.group(2).strip()

    def run(self, question: str) -> RunResult:
        transcript = ""
        steps: list[Step] = []
        last_signature = None
        repeat_count = 0

        for index in range(1, self.max_steps + 1):
            output = self.model.generate(build_prompt(question, transcript))
            try:
                thought, action, argument = self.parse(output)
            except ValueError as exc:
                observation = f"Parser error: {exc} Follow the required two-line format."
                transcript += f"{output.strip()}\nObservation: {observation}\n"
                continue

            signature = (action.casefold(), argument.casefold())
            repeat_count = repeat_count + 1 if signature == last_signature else 0
            last_signature = signature
            if repeat_count >= 2:
                observation = "Repeated action blocked. Use a different query or finish with current evidence."
            elif action == "Search":
                observation = self.environment.search(argument)
            elif action == "Lookup":
                observation = self.environment.lookup(argument)
            else:
                steps.append(Step(index, thought, action, argument, "finished"))
                return RunResult(question, argument, "finished", steps)

            steps.append(Step(index, thought, action, argument, observation))
            transcript += (
                f"Thought: {thought}\nAction: {action}[{argument}]\n"
                f"Observation: {observation}\n"
            )
        return RunResult(question, "", "max_steps", steps)

    @staticmethod
    def save_trace(result: RunResult, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
