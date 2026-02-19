from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "openrouter"
    llm_model: str = "openai/gpt-oss-120b:free"
    llm_model_planning: str = ""
    llm_model_analysis: str = ""
    llm_model_writing: str = ""
    llm_route_fallback_enabled: bool = True
    llm_fallback_provider: str = ""
    llm_fallback_model: str = ""
    llm_second_fallback_provider: str = ""
    llm_second_fallback_model: str = ""
    llm_retry_max_attempts: int = 4
    llm_retry_base_delay_s: float = 1.0
    llm_retry_max_delay_s: float = 8.0
    openai_api_key: str = ""
    groq_api_key: str = ""
    xai_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"

    search_provider: str = "mock"
    tavily_api_key: str = ""
    serpapi_api_key: str = ""

    max_search_results: int = 5
    max_sub_questions: int = 5
    max_research_iterations: int = 5
    min_relevance_score: float = 0.8
    min_unique_source_domains: int = 2
    min_reference_count: int = 3
    quality_gate_enforce: bool = False
    max_total_tokens_per_query: int = 0
    max_seconds_per_query: float = 0.0
    adaptive_depth_enabled: bool = True
    adaptive_max_passes: int = 1
    adaptive_sub_questions_increment: int = 1
    adaptive_iterations_increment: int = 1
    reports_directory: str = "./reports"
    session_db_path: str = "./reports/sessions.db"
    api_auth_token: str = ""
    api_rate_limit_per_minute: int = 0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
