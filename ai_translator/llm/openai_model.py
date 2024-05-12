import time

import streamlit as st
from loguru import logger
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from openai.types import Completion
from openai.types.chat import ChatCompletion

from ai_translator.llm.llm_base import LLMBase


class OpenAIModel(LLMBase):
    """OpenAI模型。"""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        """模型初始化。

        Args:
            model: 使用模型版本。
            api_key: API Key。
            base_url: API Base URL。
        """
        self.model: str = model
        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)

    def make_request(self, prompt: str) -> tuple[str, bool]:
        attempts: int = 0
        while attempts < 3:
            try:
                if self.model == "gpt-3.5-turbo":
                    response: ChatCompletion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    logger.debug(response)
                    translation: str = response.choices[0].message.content.strip()
                    logger.debug(translation)
                    if "```" in translation:
                        translation = translation.replace("```json", "")
                        translation = translation.replace("```", "")
                else:
                    response: Completion = self.client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=150,
                        temperature=0,
                    )
                    translation: str = response.choices[0].text.strip()
                return translation, True
            except RateLimitError as e:
                attempts += 1
                if attempts < 3:
                    logger.warning("Rate limit reached. Waiting for 60 seconds before retrying.")
                    time.sleep(60)
                else:
                    raise e
            except APIConnectionError as e:
                logger.error("The server could not be reached")
                logger.error(e.__cause__)
            except APIStatusError as e:
                logger.error("Another non-200-range status code was received")
                logger.error(e.status_code)
                logger.debug(e.response)
            except Exception as e:
                logger.exception(e)
                raise e
        return "", False
