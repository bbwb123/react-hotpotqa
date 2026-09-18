# ReAct-HotpotQA

Reproduction and engineering implementation of **ReAct: Synergizing Reasoning and Acting in Language Models**
on multi-hop question answering. The agent interleaves language-model reasoning with Wikipedia tool calls and
is evaluated with HotpotQA-style Exact Match and token-level F1.

Paper: https://arxiv.org/abs/2210.03629  
Original implementation: https://github.com/ysymyth/ReAct

## What is implemented

- A from-scratch ReAct loop: `Thought -> Action -> Observation`;
- three paper-style actions: `Search[entity]`, `Lookup[keyword]`, and `Finish[answer]`;
- a Wikipedia environment backed by the MediaWiki API;
- bounded trajectories, parser error recovery, repeated-action detection, and JSONL tracing;
- HotpotQA-compatible Exact Match and token-level F1 evaluation;
- an offline mock-model test suite, so agent control flow can be tested without consuming API quota.

This repository intentionally implements the control loop rather than wrapping a prebuilt agent executor. That
makes the interaction between the LLM, tool router, environment state, and trajectory explicit.

## Architecture

```text
question
   |
   v
ReActAgent <---- trajectory / observations
   |                         ^
   | Action                  | Observation
   v                         |
Tool parser ---> WikipediaEnvironment
   |
   +---- Finish ---> answer + trace
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Put your API key in `.env`:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

## Run one question

```bash
python -m react_qa.cli ask \
  "What nationality was the director of the film that starred Joaquin Phoenix as Freddie Quell?"
```

The complete reasoning/tool trajectory is saved to `runs/` by default.

## Run the evaluation set

```bash
python -m react_qa.cli evaluate --data data/hotpotqa_sample.jsonl
```

The bundled file is a small schema-compatible smoke-test set, not a claim of reproducing the paper's full test
score. For a formal experiment, download HotpotQA and pass a JSON/JSONL file containing `question` and `answer`.

## Test the agent loop without an API key

```bash
python -m unittest discover -s tests -v
```

The tests use a scripted LLM and an in-memory Wikipedia environment to verify multi-step tool use, action parsing,
termination, and EM/F1 scoring.

## Example trajectory

```text
Thought 1: I should identify the film and then its director.
Action 1: Search[Freddie Quell]
Observation 1: Freddie Quell is a character in The Master ... directed by Paul Thomas Anderson.
Thought 2: I need the director's nationality.
Action 2: Search[Paul Thomas Anderson]
Observation 2: Paul Thomas Anderson is an American filmmaker.
Thought 3: The answer is American.
Action 3: Finish[American]
```

## Repository layout

```text
react_qa/
  agent.py          ReAct control loop and trajectory records
  llm.py            OpenAI Responses API adapter
  tools.py          Wikipedia Search/Lookup environment
  prompts.py        ReAct instruction and few-shot prompt
  metrics.py        HotpotQA EM/F1
  evaluator.py      dataset runner and aggregate metrics
  cli.py            command-line interface
data/                sample evaluation questions
tests/               offline tests
```

## Limitations and next steps

- Wikipedia content can change, so exact trajectories are not deterministic.
- Text action parsing follows the original ReAct presentation; structured function calling is a useful ablation.
- The included questions only validate the pipeline. Full-paper reproduction requires the official dataset,
  prompt examples, model settings, and multiple-run statistical reporting.
- Useful extensions include trajectory memory, retrieval caching, uncertainty-aware stopping, and comparing
  ReAct against direct-answer and chain-of-thought baselines.

## Reference

```bibtex
@inproceedings{yao2023react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
