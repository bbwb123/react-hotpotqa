import unittest

from react_qa.agent import ReActAgent
from react_qa.metrics import exact_match, token_f1


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, prompt):
        return next(self.outputs)


class MemoryEnvironment:
    def __init__(self):
        self.calls = []

    def search(self, query):
        self.calls.append(("search", query))
        facts = {
            "Freddie Quell": "Freddie Quell is the protagonist of The Master, directed by Paul Thomas Anderson.",
            "Paul Thomas Anderson": "Paul Thomas Anderson is an American filmmaker.",
        }
        return facts.get(query, "not found")

    def lookup(self, keyword):
        self.calls.append(("lookup", keyword))
        return "lookup result"


class ReActAgentTests(unittest.TestCase):
    def test_multihop_tool_trajectory(self):
        model = ScriptedModel([
            "Thought: identify the film and director\nAction: Search[Freddie Quell]",
            "Thought: find the director's nationality\nAction: Search[Paul Thomas Anderson]",
            "Thought: evidence says American\nAction: Finish[American]",
        ])
        environment = MemoryEnvironment()
        result = ReActAgent(model, environment).run("question")
        self.assertEqual(result.answer, "American")
        self.assertEqual(result.status, "finished")
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(environment.calls[1], ("search", "Paul Thomas Anderson"))

    def test_action_parser(self):
        parsed = ReActAgent.parse("Thought: verify a fact\nAction: Lookup[nationality]")
        self.assertEqual(parsed, ("verify a fact", "Lookup", "nationality"))

    def test_hotpotqa_metrics(self):
        self.assertEqual(exact_match("The United States.", "United States"), 1.0)
        self.assertAlmostEqual(token_f1("American filmmaker", "American director"), 0.5)


if __name__ == "__main__":
    unittest.main()
