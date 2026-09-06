import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "render_glm_references",
    Path(__file__).resolve().parents[1] / "gguf-tools/quality-testing/render_glm_references.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class GLMReferenceRenderTest(unittest.TestCase):
    def response(self, reasoning="", tokens=0):
        return {"model": "z-ai/glm-5.3-flash",
                "choices": [{"message": {"content": "Answer", "reasoning": reasoning}}],
                "usage": {"completion_tokens_details": {"reasoning_tokens": tokens}}}

    def test_preserves_reasoning_and_whitespace(self):
        self.assertEqual(renderer.render("Question\n", self.response("Think\n", 2), "low"),
                         "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Question\n"
                         "<|assistant|><think>Think\n</think>")

    def test_empty_reasoning(self):
        self.assertTrue(renderer.render("Q", self.response(), "high").endswith("<think></think>"))

    def test_missing_reasoning_rejected(self):
        with self.assertRaises(ValueError):
            renderer.render("Q", self.response(tokens=10), "low")

    def test_other_model_rejected(self):
        response = self.response()
        response["model"] = "another-model"
        with self.assertRaises(ValueError):
            renderer.render("Q", response, "low")

    def test_unsupported_effort_rejected(self):
        with self.assertRaises(ValueError):
            renderer.render("Q", self.response(), "none")


if __name__ == "__main__":
    unittest.main()
