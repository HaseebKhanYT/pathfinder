"""Offline tests for the query layer. Fake LLMs only: no API calls, no data/ needed.

Run: .venv/bin/python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pathfinder.criteria.engine import load_rules
from pathfinder.query import fallback as fb
from pathfinder.query.catalog import build_catalog
from pathfinder.query.run import format_rule, run_query
from pathfinder.query.translate import RuleDraft, translate
from pathfinder.query.validate import validate_rule

RECORDS = {
    "A": {"organ": "colon", "lymph_nodes_positive": 2, "pN_base": "pN1b", "pT": "pT3", "age_min": 70,
          "lymphovascular_invasion": "present", "involved_margins": []},
    "B": {"organ": "colon", "lymph_nodes_positive": 0, "pN_base": "pN0", "pT": "pT4a", "age_min": 50,
          "lymphovascular_invasion": "not_identified", "involved_margins": []},
    "C": {"organ": "breast", "lymph_nodes_positive": 1, "pN_base": "pN1a", "pT": "pT2", "age_min": 40,
          "lymphovascular_invasion": "present", "involved_margins": ["deep"]},
}
TEXTS = {
    "A": "FINAL DIAGNOSIS: Adenocarcinoma. Mismatch repair proteins: loss of MLH1 and PMS2 expression.",
    "B": "FINAL DIAGNOSIS: Adenocarcinoma. Mismatch repair proteins: intact expression.",
    "C": "Breast report.",
}


class FakeLLM:
    """Returns queued RuleDraft/Verdict objects; records every call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return {"parsed": self.outputs.pop(0), "raw": None}

    def batch_as_completed(self, inputs, config=None, return_exceptions=False):
        for i, m in enumerate(inputs):
            yield i, self.invoke(m)


def draft(rule, answer="ids", by=None, uncovered=None):
    return RuleDraft(interpretation="test", rule_json=json.dumps(rule), answer=answer,
                     breakdown_by=by, uncovered=uncovered)


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.cat = build_catalog(RECORDS)

    def test_existing_criteria_rules_are_valid(self):
        cat = build_catalog()  # schema only
        for cid, spec in load_rules().items():
            self.assertEqual(validate_rule(spec["rule"], cat), [], cid)

    def test_rejects_unknown_field_bad_enum_bad_op_and_unseen_text(self):
        self.assertIn("unknown field", validate_rule(["tumor_size", "gt", 20], self.cat)[0])
        self.assertIn("not an allowed value", validate_rule(["organ", "eq", "liver"], self.cat)[0])
        self.assertIn("not valid for list", validate_rule(["involved_margins", "eq", "deep"], self.cat)[0])
        self.assertIn("never occurs", validate_rule(["pT", "eq", "T4"], self.cat)[0])
        self.assertIn("numeric", validate_rule(["age_min", "gte", "70"], self.cat)[0])
        self.assertEqual(validate_rule(["pT", "startswith", "pT4"], self.cat), [])


class TestTranslate(unittest.TestCase):
    def test_repair_loop_sends_errors_back_and_accepts_fix(self):
        cat = build_catalog(RECORDS)
        llm = FakeLLM([draft(["organ", "eq", "Colon"]), draft(["organ", "eq", "colon"])])
        t = translate("colon cancer", cat, "fake", llm=llm)
        self.assertTrue(t.ok)
        self.assertEqual(len(t.attempts), 2)
        self.assertIn("not an allowed value", llm.calls[1][-1].content)  # error fed back

    def test_gives_up_after_max_repairs(self):
        cat = build_catalog(RECORDS)
        llm = FakeLLM([draft(["nope", "eq", 1])] * 3)
        t = translate("x", cat, "fake", max_repairs=2, llm=llm)
        self.assertFalse(t.ok)
        self.assertEqual(len(t.attempts), 3)

    def test_examples_can_be_excluded(self):
        from pathfinder.query.translate import system_prompt
        cat = build_catalog()
        self.assertIn(load_rules()["c08"]["text"], system_prompt(cat))
        self.assertNotIn(load_rules()["c08"]["text"], system_prompt(cat, {"c08"}))


class TestRunAndFallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(fb, "CACHE", Path(self.tmp.name) / "cache.jsonl")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_count_and_breakdown_without_api(self):
        r = run_query("q", RECORDS, TEXTS, "fake", lambda e: False, 10,
                      rule=["lymph_nodes_positive", "gte", 1], answer_type="breakdown", breakdown_by="organ")
        self.assertEqual(r.ids, ["A", "C"])
        self.assertEqual(dict(r.breakdown), {"colon": 1, "breast": 1})

    def test_fallback_declined_makes_no_calls_and_gives_no_answer(self):
        tr = FakeLLM([draft(["organ", "eq", "colon"], uncovered="Is MMR deficient?")])
        fl = FakeLLM([])
        r = run_query("colon MMR deficient", RECORDS, TEXTS, "fake", lambda e: False, 10,
                      translator_llm=tr, fallback_llm=fl)
        self.assertEqual(fl.calls, [])
        self.assertEqual(r.ids, [])
        self.assertEqual(r.rule_ids, ["A", "B"])
        self.assertIn("not confirmed", r.fallback.skipped_reason)

    def test_fallback_cap_refuses_before_asking(self):
        asked = []
        tr = FakeLLM([draft(["organ", "eq", "colon"], uncovered="Is MMR deficient?")])
        r = run_query("q", RECORDS, TEXTS, "fake", lambda e: asked.append(e) or True, 1,
                      translator_llm=tr, fallback_llm=FakeLLM([]))
        self.assertEqual(asked, [])
        self.assertIn("above the cap", r.fallback.skipped_reason)

    def test_fallback_keeps_only_verified_quotes_and_caches(self):
        tr = FakeLLM([draft(["organ", "eq", "colon"], uncovered="Is MMR deficient?")])
        fl = FakeLLM([fb.Verdict(matches=True, evidence_quote="loss of MLH1 and PMS2 expression"),
                      fb.Verdict(matches=True, evidence_quote="MSH6 loss")])  # invented quote for B
        r = run_query("q", RECORDS, TEXTS, "fake", lambda e: True, 10, translator_llm=tr, fallback_llm=fl)
        self.assertEqual(r.ids, ["A"])
        self.assertEqual(r.fallback.unverified, ["B"])
        # Second run: everything cached, so no confirmation and no calls.
        tr2 = FakeLLM([draft(["organ", "eq", "colon"], uncovered="Is MMR deficient?")])
        r2 = run_query("q", RECORDS, TEXTS, "fake", lambda e: self.fail("asked"), 10,
                       translator_llm=tr2, fallback_llm=FakeLLM([]))
        self.assertEqual(r2.ids, ["A"])

    def test_format_rule(self):
        s = format_rule({"all": [["organ", "eq", "colon"], {"not": ["pT", "startswith", "pT4"]}]})
        self.assertEqual(s, 'ALL of:\n  ["organ", "eq", "colon"]\n  NOT\n    ["pT", "startswith", "pT4"]')


if __name__ == "__main__":
    unittest.main()


class TestReviewFixes(unittest.TestCase):
    def setUp(self):
        self.cat = build_catalog(RECORDS)
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(fb, "CACHE", Path(self.tmp.name) / "cache.jsonl")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_short_or_partial_quotes_are_not_evidence(self):
        text = "Specimen: right colon. No lymphovascular invasion. Mismatch repair: loss of MLH1."
        for q in ["", "a", "No", "colon", "loss of"]:
            self.assertFalse(fb.quote_in_text(q, text), q)
        self.assertTrue(fb.quote_in_text("mismatch repair:  LOSS of MLH1", text))
        self.assertFalse(fb.quote_in_text("loss of MLH1 and PMS2", text))

    def test_failed_call_is_reported_and_count_is_lower_bound(self):
        class Flaky(FakeLLM):
            def batch_as_completed(self, inputs, config=None, return_exceptions=False):
                yield 0, {"parsed": fb.Verdict(matches=True, evidence_quote="loss of MLH1 and PMS2 expression"), "raw": None}
                yield 1, RuntimeError("boom")
        r = run_query("q", RECORDS, TEXTS, "fake", lambda e: True, 10, rule=["organ", "eq", "colon"],
                      uncovered="Is MMR deficient?", fallback_llm=Flaky([]))
        self.assertEqual(r.fallback.failed, ["B"])
        self.assertEqual(r.fallback.usage["calls"], 2)
        self.assertFalse(r.complete)

    def test_validator_edge_cases(self):
        self.assertIn("non-empty", validate_rule(["pT", "startswith", []], self.cat)[0])
        self.assertIn("not part of any value", validate_rule(["involved_margins", "contains", "Deep"], self.cat)[0])
        self.assertEqual(validate_rule(["involved_margins", "contains", "deep"], self.cat), [])
        r = run_query("q", RECORDS, TEXTS, "fake", lambda e: False, 10, rule=["organ", "eq", "colon"],
                      answer_type="breakdown")
        self.assertIn("--by", r.error)

    def test_cache_key_depends_on_effort(self):
        self.assertNotEqual(fb._key("q", "m", "A", "low"), fb._key("q", "m", "A", "high"))
        self.assertEqual(fb._key("Is  MMR deficient?", "m", "A"), fb._key("is mmr deficient?", "m", "A"))
