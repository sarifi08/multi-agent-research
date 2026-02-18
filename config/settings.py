"""Configuration for Multi-Agent Research System."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # APIs
    openai_api_key: str = Field(..., description="Powers agent brains")
    tavily_api_key: str = Field(..., description="Powers web search")

    # Model — gpt-4o for quality, gpt-3.5-turbo for cost savings
    llm_model: str = Field(default="gpt-4o")
    max_output_tokens: int = Field(default=2000)

    # Agent behavior
    max_retries: int = Field(default=2, description="Before giving up and passing empty to Analyst")
    max_parallel_searches: int = Field(default=3, description="API rate limit protection")
    max_search_results: int = Field(default=5)

    # Monitoring
    enable_tracking: bool = Field(default=True)


def get_settings() -> Settings:
    return Settings()
