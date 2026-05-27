"""
config.py
=========
Centralised settings loaded from environment variables / .env file.
Uses Google Application Default Credentials — no GOOGLE_API_KEY required.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Google Cloud ────────────────────────────────────────────────────────
    google_cloud_project: str
    google_cloud_location: str = "us-central1"

    # ── Model selection ─────────────────────────────────────────────────────
    # Supervisor: fast routing decisions
    supervisor_model: str = "gemini-2.0-flash"
    # Sub-agents: deep analysis & research
    subagent_model: str = "gemini-2.5-pro"

    # ── Sub-agent databases (READ-ONLY) ─────────────────────────────────────
    kpi_db_dsn: str
    kpi_db_schemas: str = "kpi_config,kpi_values"
    
    iot_db_dsn: str
    iot_db_schemas: str = "public"
    
    asset_db_dsn: str
    asset_db_schemas: str = "public"
    
    @property
    def kpi_schemas_list(self) -> list[str]:
        return [s.strip() for s in self.kpi_db_schemas.split(",") if s.strip()]
        
    @property
    def iot_schemas_list(self) -> list[str]:
        return [s.strip() for s in self.iot_db_schemas.split(",") if s.strip()]
        
    @property
    def asset_schemas_list(self) -> list[str]:
        return [s.strip() for s in self.asset_db_schemas.split(",") if s.strip()]

    # ── Conversation / session database (READ + WRITE) ───────────────────────
    # Used for LangGraph checkpointing and conversation history
    conv_db_dsn: str

    # ── API server ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Agent behaviour ─────────────────────────────────────────────────────
    # Maximum rows returned from any single DB query inside an agent
    db_max_rows: int = 500
    # Maximum recursion depth for cross-agent calls (prevents infinite loops)
    max_agent_call_depth: int = 2


# Singleton — import this everywhere
settings = Settings()
