from typing import Any

import requests
import simplejson
from requests import Response

from ai_translator.llm.llm_base import LLMBase


class GLMModel(LLMBase):
    """GLM模型。"""

    def __init__(self, model_url: str, timeout: int) -> None:
        """模型初始化。

        Args:
            model_url: 模型URL。
            timeout: 超时秒数。
        """
        self.model_url: str = model_url
        self.timeout: int = timeout

    def make_request(self, prompt) -> tuple[str, bool]:
        try:
            payload: dict[str, Any] = {"prompt": prompt, "history": []}
            response: Response = requests.post(self.model_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            response_dict: dict[str, Any] = response.json()
            translation: str = response_dict["response"]
            return translation, True
        except requests.exceptions.Timeout as e:
            raise Exception(f"请求超时：{e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求异常：{e}")
        except simplejson.JSONDecodeError:
            raise Exception("Error: response is not valid JSON format.")
        except Exception as e:
            raise Exception(f"发生了未知错误：{e}")
