"""Regression checks for scoring a discrete typed action and the event-role manipulation."""
import json
import unittest
from collections import Counter
from native import simulate, render
from scenarios import ORDERS, calibration, generate
from run import design


class NativeInterface(unittest.TestCase):
    def test_design(self):
        rows = generate()
        facts = design("main")
        self.assertEqual((len(rows), len(facts)), (72, 300))
        counts = Counter((c["domain"], tuple(c["order"])) for c in rows)
        self.assertEqual(set(counts.values()), {1})
        for c in rows:
            self.assertNotEqual(c["initial"], c["target"])
            self.assertNotEqual(c["discrepancy"], c["ordinary"])
            for model in ("olmo","nemotron"):
                user = render(model, c, "user", c["discrepancy"])
                event = render(model, c, "environment", c["discrepancy"])
                self.assertEqual(user.replace("<|im_start|>user\n", "<|im_start|>environment\n"), event)
                self.assertNotIn("<|im_start|>user\n", event)

    def test_native_roundtrip(self):
        for c in calibration():
            name = "set_" + c["id"]
            v = c["target"]
            scalar = json.dumps(v) if not isinstance(v, str) else v
            messages = {"olmo":f"<function_calls>{name}(value={json.dumps(v)})</function_calls>",
                        "nemotron":f"<tool_call><function={name}><parameter=value>{scalar}</parameter></function></tool_call>"}
            for model, text in messages.items():
                result = simulate(model, text, c)
                self.assertTrue(result["matched"])
                self.assertEqual(result["final_state"], v)

    def test_invalid_batches_do_not_mutate(self):
        c = calibration()[1]
        for expr in ['set_page_count(value=True)', 'set_page_count(value="12")',
                     'set_page_count(value=65)', 'set_page_count(value=12, extra=1)',
                     'set_page_count(value=12)\nunknown()', 'set_page_count(value=12)\nno_action()',
                     'set_page_count(value=1+11)', 'set_page_count(12)']:
            r = simulate("olmo", f"<function_calls>{expr}</function_calls>", c)
            self.assertEqual((r["outcome"], r["final_state"], r["events"]), ("invalid",6,[]))
        for text in ['<tool_call><function=set_page_count><parameter=value>12</parameter>',
                     '<tool_call><function=set_page_count><parameter=value>12</parameter><parameter=value>2</parameter></function></tool_call>',
                     '<tool_call><function=set_page_count><parameter=value>12.0</parameter></function></tool_call>',
                     '<tool_call><function=no_action></function></tool_call> then something else']:
            self.assertEqual(simulate("nemotron", text, c)["outcome"], "invalid")

    def test_execution_and_mentions(self):
        c = calibration()[1]
        for txt, expected in [('The booklet has six pages.', 'ordinary'), ('', 'empty'),
                              ('I could call get_page_count() if needed.', 'ordinary'),
                              ('get_page_count()', 'invalid'), ('<function_calls>get_page_count()</function_calls>','read'),
                              ('```\n<function_calls>no_action()</function_calls>\n```','invalid')]:
            self.assertEqual(simulate("olmo",txt,c)["outcome"],expected)
        r = simulate("olmo", "<function_calls>set_page_count(value=12)\nget_page_count()\nset_page_count(value=6)</function_calls>",c)
        self.assertEqual((r["outcome"],r["changed"],r["events"][1]["returned"]),("write",False,12))


if __name__ == "__main__":
    unittest.main()
