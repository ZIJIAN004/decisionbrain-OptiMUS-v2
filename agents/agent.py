from typing import Dict, Optional, Union, List
import openai
from openai import Client, OpenAI

# Every agent now shares one OpenAI-compatible client (utils/misc.py:
# get_llm_client), so the Mistral branch that used to live in llm_call is gone
# along with its SDK import -- keeping it would make the whole pipeline depend
# on a package it no longer calls.


class Agent:
    def __init__(self, name, description, client, llm="gpt-3.5-turbo", **kwargs):
        self.name = name
        self.description = description
        self.client = client
        self.system_prompt = "You're a helpful assistant."
        self.kwargs = kwargs
        self.llm = llm

    def llm_call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List] = None,
        seed: int = 10,
    ) -> str:
        model = self.llm
        # make sure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)
        # make sure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if not prompt is None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed,
            )
        except (TypeError, openai.BadRequestError):
            # Not every OpenAI-compatible endpoint accepts `seed`. Losing it
            # costs reproducibility of individual replies, so it is dropped only
            # when the endpoint refuses it, never pre-emptively. A 400 raised for
            # any other reason will simply raise again on this second attempt.
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )

        return completion.choices[0].message.content

    def generate_reply(
        self,
        task: str,
        state: Dict,
        sender: "Agent",
    ) -> (str, Dict):
        return (
            "This is a reply from the agent. REPLY NOT IMPLEMENTED! Terminate the whole process!",
            state,
        )
