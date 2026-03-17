"""Tests for TeacherInferenceEngineClient logprob extraction."""

import pytest
import torch
from skyrl_train.inference_engines.teacher_engine_client import (
    TeacherInferenceEngineClient,
    TeacherScoringOutput,
)


class TestExtractResponseLogprobs:
    """Test _extract_response_logprobs with mock vLLM output format."""

    def _make_client(self):
        """Create a client with no real engines (for testing extraction only)."""
        client = TeacherInferenceEngineClient.__new__(TeacherInferenceEngineClient)
        client.inference_engines = []
        client.top_k_logprobs = 3
        client._engine_idx = 0
        return client

    def test_basic_extraction(self):
        """Test extraction from well-formed prompt_logprobs."""
        client = self._make_client()

        # Simulate vLLM output: 1 sample, prompt_len=2, response_len=3
        # prompt_logprobs covers positions 0..4 (prompt[0,1] + response[0,1,2])
        prompt_logprobs_batch = [[
            None,                           # position 0 (first prompt token, always None)
            {10: -1.0, 20: -2.0, 30: -3.0},  # position 1 (second prompt token)
            {100: -0.5, 200: -1.5, 300: -2.5},  # position 2 (response token 0)
            {101: -0.3, 201: -1.8, 301: -3.0},  # position 3 (response token 1)
            {102: -0.1, 202: -2.0, 302: -4.0},  # position 4 (response token 2)
        ]]

        output = {
            "responses": ["test"],
            "response_ids": [[999]],  # dummy
            "stop_reasons": ["length"],
            "response_logprobs": None,
            "prompt_logprobs": prompt_logprobs_batch,
        }

        result = client._extract_response_logprobs(
            output=output,
            prompt_lengths=[2],
            response_token_ids=[[100, 101, 102]],
            k=3,
        )

        assert isinstance(result, TeacherScoringOutput)
        assert result.top_k_logprobs.shape == (1, 3, 3)
        assert result.top_k_indices.shape == (1, 3, 3)
        assert result.chosen_token_logprobs.shape == (1, 3)

        # Check response token 0 (position 2): top-K should be sorted by logprob desc
        assert result.top_k_logprobs[0, 0, 0].item() == pytest.approx(-0.5)   # token 100
        assert result.top_k_logprobs[0, 0, 1].item() == pytest.approx(-1.5)   # token 200
        assert result.top_k_logprobs[0, 0, 2].item() == pytest.approx(-2.5)   # token 300
        assert result.top_k_indices[0, 0, 0].item() == 100
        assert result.top_k_indices[0, 0, 1].item() == 200

        # Check chosen token logprobs
        assert result.chosen_token_logprobs[0, 0].item() == pytest.approx(-0.5)  # token 100 at pos 2
        assert result.chosen_token_logprobs[0, 1].item() == pytest.approx(-0.3)  # token 101 at pos 3
        assert result.chosen_token_logprobs[0, 2].item() == pytest.approx(-0.1)  # token 102 at pos 4

    def test_batch_extraction(self):
        """Test extraction with multiple samples in batch."""
        client = self._make_client()

        prompt_logprobs_batch = [
            # Sample 0: prompt_len=1, response_len=2
            [
                None,
                {50: -1.0, 60: -2.0},   # response token 0
                {51: -0.5, 61: -1.5},   # response token 1
            ],
            # Sample 1: prompt_len=2, response_len=1
            [
                None,
                {70: -0.8, 80: -1.2},
                {71: -0.3, 81: -0.9},   # response token 0
            ],
        ]

        output = {
            "responses": ["a", "b"],
            "response_ids": [[999], [998]],
            "stop_reasons": ["length", "length"],
            "response_logprobs": None,
            "prompt_logprobs": prompt_logprobs_batch,
        }

        result = client._extract_response_logprobs(
            output=output,
            prompt_lengths=[1, 2],
            response_token_ids=[[50, 51], [71]],
            k=3,
        )

        assert result.top_k_logprobs.shape == (2, 2, 3)  # max_response_len=2

        # Sample 0, response token 0
        assert result.chosen_token_logprobs[0, 0].item() == pytest.approx(-1.0)
        # Sample 1, response token 0
        assert result.chosen_token_logprobs[1, 0].item() == pytest.approx(-0.3)

    def test_padding_for_unequal_response_lengths(self):
        """Test that shorter responses are padded with -inf."""
        client = self._make_client()

        prompt_logprobs_batch = [
            [None, {10: -1.0}],              # prompt_len=1, response_len=1
            [None, {20: -0.5}, {21: -0.3}],  # prompt_len=1, response_len=2
        ]

        output = {
            "responses": ["a", "bb"],
            "response_ids": [[10], [20, 21]],
            "stop_reasons": ["eos", "eos"],
            "response_logprobs": None,
            "prompt_logprobs": prompt_logprobs_batch,
        }

        result = client._extract_response_logprobs(
            output=output,
            prompt_lengths=[1, 1],
            response_token_ids=[[10], [20, 21]],
            k=3,
        )

        assert result.top_k_logprobs.shape == (2, 2, 3)  # max_response_len=2
        # Sample 0, position 1 should be -inf (padded)
        assert result.top_k_logprobs[0, 1, 0].item() == float("-inf")

    def test_no_prompt_logprobs(self):
        """Test graceful handling when prompt_logprobs is None."""
        client = self._make_client()

        output = {
            "responses": ["test"],
            "response_ids": [[999]],
            "stop_reasons": ["length"],
            "response_logprobs": None,
            "prompt_logprobs": None,
        }

        result = client._extract_response_logprobs(
            output=output,
            prompt_lengths=[5],
            response_token_ids=[[100, 200]],
            k=3,
        )

        # Should return zero tensors (no data extracted)
        assert result.top_k_logprobs.shape == (1, 2, 3)
        assert (result.top_k_logprobs == float("-inf")).all()
        assert (result.chosen_token_logprobs == 0).all()

    def test_chosen_token_not_in_top_k(self):
        """Test when the chosen token isn't in the top-K alternatives."""
        client = self._make_client()

        prompt_logprobs_batch = [[
            None,
            # Top-3 are tokens 10, 20, 30 — but chosen token is 99 (not in dict)
            {10: -0.5, 20: -1.0, 30: -1.5},
        ]]

        output = {
            "responses": ["test"],
            "response_ids": [[999]],
            "stop_reasons": ["length"],
            "response_logprobs": None,
            "prompt_logprobs": prompt_logprobs_batch,
        }

        result = client._extract_response_logprobs(
            output=output,
            prompt_lengths=[1],
            response_token_ids=[[99]],  # token 99 not in dict
            k=3,
        )

        # chosen_token_logprobs should be 0 (default) since token not found
        assert result.chosen_token_logprobs[0, 0].item() == 0.0
        # But top-K should still be populated
        assert result.top_k_logprobs[0, 0, 0].item() == pytest.approx(-0.5)
