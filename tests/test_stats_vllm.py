import conftest_paths  # noqa: F401
import unittest
import stats


class TestVllmMetrics(unittest.TestCase):
    def test_parses_vllm_prompt_and_gen_counters(self):
        text = (
            '# HELP vllm:prompt_tokens_total ...\n'
            'vllm:prompt_tokens_total{model_name="Qwen/Qwen3-8B"} 1234\n'
            'vllm:generation_tokens_total{model_name="Qwen/Qwen3-8B"} 5678\n'
        )
        p, g = stats.vllm_token_totals(stats._parse_metrics(text))
        self.assertEqual((p, g), (1234.0, 5678.0))

    def test_missing_metrics_zero(self):
        p, g = stats.vllm_token_totals({})
        self.assertEqual((p, g), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
