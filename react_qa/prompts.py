SYSTEM_PROMPT = """You are a question-answering agent. Solve the question by interleaving reasoning and actions.

At each step output exactly two lines:
Thought: a concise statement of what you know and what you need next
Action: one of Search[entity], Lookup[keyword], or Finish[answer]

Search[entity] retrieves a Wikipedia page summary and makes that page active.
Lookup[keyword] retrieves the next sentence containing keyword from the active page.
Finish[answer] ends the task. Use tools to verify multi-hop facts before finishing.
Do not invent observations. Do not output more than one action.

Example:
Question: What nationality was the director of the film that starred Joaquin Phoenix as Freddie Quell?
Thought: I should identify the film associated with Freddie Quell.
Action: Search[Freddie Quell]
Observation: Freddie Quell is the protagonist of The Master, a film directed by Paul Thomas Anderson.
Thought: I need the director's nationality.
Action: Search[Paul Thomas Anderson]
Observation: Paul Thomas Anderson is an American filmmaker.
Thought: The retrieved evidence states his nationality.
Action: Finish[American]
"""


def build_prompt(question: str, transcript: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}\n"
    if transcript:
        prompt += transcript.rstrip() + "\n"
    return prompt
