from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .agent import ReActAgent
from .evaluator import evaluate, load_examples
from .llm import OpenAIResponsesModel
from .tools import WikipediaEnvironment


def make_agent(max_steps: int) -> ReActAgent:
    return ReActAgent(OpenAIResponsesModel(), WikipediaEnvironment(), max_steps=max_steps)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="ReAct agent for HotpotQA-style questions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask = subparsers.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--max-steps", type=int, default=8)
    ask.add_argument("--run-dir", default="runs")
    run_eval = subparsers.add_parser("evaluate")
    run_eval.add_argument("--data", required=True)
    run_eval.add_argument("--max-steps", type=int, default=8)
    run_eval.add_argument("--run-dir", default="runs")
    args = parser.parse_args()

    agent = make_agent(args.max_steps)
    if args.command == "ask":
        result = agent.run(args.question)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        agent.save_trace(result, Path(args.run_dir) / f"trace_{stamp}.json")
        for step in result.steps:
            print(f"Thought {step.index}: {step.thought}")
            print(f"Action {step.index}: {step.action}[{step.argument}]")
            print(f"Observation {step.index}: {step.observation}\n")
        print(f"Answer: {result.answer or '[no answer]'}")
    else:
        summary = evaluate(agent, load_examples(args.data), args.run_dir)
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
