import asyncio
import json
import unittest

import aiohttp
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text

class TestSessionControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "flashinfer",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_session_control(self, gen_len=12):
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
            "A brief history about that city is",
        ]
        tokenizer = get_tokenizer(self.model)
        chunks_ids = [tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i] and chunks_ids[i][0] == tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        # 1. using session control
        requests.p