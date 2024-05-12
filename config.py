from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseModel):
    """OpenAI配置信息。"""

    base_url: Optional[HttpUrl] = Field(title="API Base URL", default=None)
    api_key: str = Field(title="API Key")
    model: str = Field(title="默认模型", default="gpt-3.5-turbo")


class GLMConfig(BaseModel):
    """GLM配置信息。"""

    base_url: Optional[HttpUrl] = Field(title="模型URL")
    time_out: int = Field(title="API超时秒数限制")


class Config(BaseSettings):
    """配置信息。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        env_nested_delimiter="__",
    )
    llm_type: Literal["OpenAIModel", "GLMModel"] = Field(title="模型类型")
    openai: Optional[OpenAIConfig] = Field(title="OpenAI配置信息", default=None)
    glm: Optional[GLMConfig] = Field(title="GLM配置信息", default=None)
