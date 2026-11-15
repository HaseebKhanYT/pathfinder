"""Offline tests for uploaded datasets and extraction progress. No API calls, no data/ needed.

Run: .venv/bin/python -m unittest discover -s tests
"""

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pathfinder import datasets
from pathfinder.datasets import dataset_name, parse_reports, save_dataset, split_path
from pathfinder.extraction import extractor


def jsonl(*rows) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


A = {"id": "R1", "report_text": "Report one."}
B = {"id": "R2", "report_text": "Report two.", "labels": {"organ": "colon"}}


class ParseReports(unittest.TestCase):
    def test_valid_rows_keep_extra_keys_and_skip_blank_lines(self):
        raw = b"\xef\xbb\xbf" + jsonl(A) + b"\n  \n" + jsonl(B)
        self.assertEqual(parse_reports(raw), [A, B])

    def test_errors_name_the_line(self):
        cases = {
            b"not json\n": "line 1 is not valid JSON",
            jsonl(A) + b"[1, 2]\n": "line 2 is not a JSON object",
            jsonl({"report_text": "x"}): 'line 1 has no "id"',
            jsonl({"id": "R1", "report_text": "  "}): 'line 1 (R1) has no "report_text"',
            jsonl(A, A): "line 2 repeats the id R1",
            b"\n\n": "no reports",
            b"\xff\xfe": "not UTF-8",
        }
        for raw, msg in cases.items():
            with self.subTest(msg=msg), self.assertRaisesRegex(ValueError, re.escape(msg)):
                parse_reports(raw)


class Naming(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name, value in (("BUILTIN_DIR", self.root / "builtin"), ("DATASETS_DIR", self.root / "datasets")):
            patcher = mock.patch.object(datasets, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        (self.root / "builtin").mkdir()
        (self.root / "builtin" / "test.jsonl").write_bytes(jsonl(A, B))

    def test_slug_is_safe_for_cache_file_names(self):
        self.assertEqual(dataset_name("My Reports v2.final.jsonl", [A]), "my-reports-v2-final")
        self.assertEqual(dataset_name("###.jsonl", [A]), "upload")

    def test_builtin_name_reuses_split_only_when_content_matches(self):
        self.assertEqual(dataset_name("test.jsonl", [A, B]), "test")
        self.assertEqual(dataset_name("TEST.jsonl", [A]), "test-upload")

    def test_split_path_and_save(self):
        self.assertEqual(split_path("test"), self.root / "builtin" / "test.jsonl")
        path = save_dataset("mine", [A, B])
        self.assertEqual(path, self.root / "datasets" / "mine.jsonl")
        self.assertEqual(parse_reports(path.read_bytes()), [A, B])

    def test_save_never_writes_a_builtin_split(self):
        before = (self.root / "builtin" / "test.jsonl").read_bytes()
        save_dataset("test", [A])
        self.assertEqual((self.root / "builtin" / "test.jsonl").read_bytes(), before)
        self.assertFalse((self.root / "datasets").exists())


class ProviderRouting(unittest.TestCase):
    def test_bare_ids_go_to_openai_and_org_ids_to_nebius(self):
        self.assertEqual(extractor.provider_key("gpt-5.4-mini"), "OPENAI_API_KEY")
        self.assertEqual(extractor.provider_key("zai-org/GLM-5.3-Flash"), "NEBIUS_API_KEY")
        self.assertEqual(extractor.provider_key("openai/gpt-oss-120b"), "NEBIUS_API_KEY")

    def test_openai_client_has_no_nebius_base_url(self):
        env = {"OPENAI_API_KEY": "sk-test", "NEBIUS_API_KEY": "nb-test"}
        with mock.patch.dict("os.environ", env), mock.patch.object(extractor, "load_dotenv"):
            oa = extractor.build_chat_model("gpt-5.4-mini")
            nb = extractor.build_chat_model("zai-org/GLM-5.3-Flash")
        self.assertNotIn("nebius", str(oa.openai_api_base or ""))
        self.assertEqual(nb.openai_api_base, extractor.NEBIUS_BASE_URL)


class ExtractProgress(unittest.TestCase):
    def test_on_progress_called_per_result_with_error_count(self):
        class FailingChain:
            def batch_as_completed(self, inputs, config=None, return_exceptions=False):
                for i, _ in enumerate(inputs):
                    yield i, RuntimeError("boom")

        calls = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(extractor, "build_chain", return_value=FailingChain()):
            cache = Path(tmp) / "x.jsonl"
            got = extractor.extract_reports([A, B], cache, on_progress=lambda *a: calls.append(a))
            lines = [json.loads(l) for l in cache.read_text().splitlines()]
        self.assertEqual(got, {})
        self.assertEqual(calls, [(1, 2, 1), (2, 2, 2)])
        self.assertTrue(all("error" in rec for rec in lines))


if __name__ == "__main__":
    unittest.main()
