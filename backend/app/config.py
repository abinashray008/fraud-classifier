"""Application settings loaded from the environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", "../.env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Jev / TypeSafe -------------------------------------------------------
    typesafe_api_key: str = Field(default="", alias="TYPESAFE_API_KEY")
    # Pin a concrete version in production so calibrated thresholds stay valid.
    jev_model: str = Field(default="jev-latest", alias="JEV_MODEL")
    jev_timeout_seconds: float = Field(default=5.0, alias="JEV_TIMEOUT_SECONDS")
    jev_max_retries: int = Field(default=1, alias="JEV_MAX_RETRIES")

    # --- Decision policy (calibrate via backend/eval/calibrate.py) ------------
    policy_t_low: float = Field(default=0.20, alias="POLICY_T_LOW")
    policy_t_high: float = Field(default=0.80, alias="POLICY_T_HIGH")
    policy_c_min: float = Field(default=0.60, alias="POLICY_C_MIN")
    policy_agent_decline_prob: float = Field(default=0.90, alias="POLICY_AGENT_DECLINE_PROB")

    # --- OTP step-up ------------------------------------------------------------
    otp_ttl_seconds: int = Field(default=300, alias="OTP_TTL_SECONDS")
    otp_max_attempts: int = Field(default=3, alias="OTP_MAX_ATTEMPTS")
    otp_code_length: int = Field(default=6, alias="OTP_CODE_LENGTH")
    # In dev mode the OTP code is returned in the API response and logged.
    otp_dev_mode: bool = Field(default=True, alias="OTP_DEV_MODE")

    # --- Investigation agent (LLM tier) ----------------------------------------
    # Provider-agnostic init_chat_model strings, e.g. "openai:gpt-5.6-luna" or
    # "anthropic:claude-...". Leave empty to disable the agent tier.
    llm_fast: str = Field(default="", alias="LLM_FAST")
    llm_powerful: str = Field(default="", alias="LLM_POWERFUL")
    agent_enabled: bool = Field(default=True, alias="AGENT_ENABLED")
    agent_tool_risk_threshold: float = Field(default=0.30, alias="AGENT_TOOL_RISK_THRESHOLD")
    agent_max_steps: int = Field(default=12, alias="AGENT_MAX_STEPS")

    # --- Server ---------------------------------------------------------------------
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # Built Vite app. Empty = auto-detect repo `frontend/dist` or `/app/frontend/dist`.
    frontend_dist: str = Field(default="", alias="FRONTEND_DIST")
    serve_frontend: bool = Field(default=True, alias="SERVE_FRONTEND")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def jev_configured(self) -> bool:
        return bool(self.typesafe_api_key)

    @property
    def agent_configured(self) -> bool:
        return self.agent_enabled and bool(self.llm_fast) and self.jev_configured


@lru_cache
def get_settings() -> Settings:
    # pydantic-settings reads .env into the Settings object only. Provider SDKs behind
    # init_chat_model (GOOGLE_API_KEY, OPENAI_API_KEY, ...) read os.environ directly, so
    # export the same files there too. Real environment variables still take precedence.
    for env_file in ENV_FILES:
        load_dotenv(env_file, override=False)
    return Settings()
