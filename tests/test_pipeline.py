import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from inference.pipeline import TranslationPipeline
from tests.test_memory import TEST_GLOSS_INPUTS, _bootstrap_assets


class PipelineTestCase(unittest.TestCase):
    def test_single_and_batch_translate(self):
        base_dir = WORKSPACE_ROOT / f"tmp_pipeline_test_{uuid.uuid4().hex}"
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            model_dir, config_path = _bootstrap_assets(base_dir)
            pipeline = TranslationPipeline(model_dir=model_dir.as_posix(), config_path=config_path.as_posix())
            single = pipeline.translate("我 昨天 买 苹果")
            batch = pipeline.batch_translate(TEST_GLOSS_INPUTS[:3])

            self.assertIsInstance(single, str)
            self.assertTrue(single is not None)
            self.assertEqual(len(batch), 3)
            self.assertTrue(all(isinstance(item, str) for item in batch))
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
