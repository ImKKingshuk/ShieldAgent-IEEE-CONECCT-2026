"""Configuration management for ShieldAgent."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Configuration for LLM backend.
    
    Supported providers:
        - openrouter: OpenRouter API (200+ models)
        - cerebras: Cerebras Cloud (fast Llama/Qwen inference)
        - google: Google AI (Gemini models)
    """
    provider: str = "openrouter"
    model: str = "meta-llama/llama-3.1-70b-instruct"
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"  # Only used for OpenRouter
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0
    
    def __post_init__(self):
        """Load API key for the configured provider if not set."""
        if self.api_key is None:
            # Load provider-specific API key from environment
            provider_key_map = {
                "openrouter": "OPENROUTER_API_KEY",
                "cerebras": "CEREBRAS_API_KEY",
                "google": "GOOGLE_AI_API_KEY",
                "google_ai": "GOOGLE_AI_API_KEY",
                "gemini": "GOOGLE_AI_API_KEY",
            }
            env_var = provider_key_map.get(self.provider.lower(), "OPENROUTER_API_KEY")
            self.api_key = os.getenv(env_var)


@dataclass
class SanitizerConfig:
    """Configuration for tool response sanitizer."""
    enabled: bool = True
    block_on_detection: bool = True
    custom_patterns: list[str] = field(default_factory=list)
    whitelist_patterns: list[str] = field(default_factory=list)


@dataclass
class AnomalyDetectorConfig:
    """Configuration for GNN-based anomaly detector."""
    enabled: bool = True
    model_path: Optional[Path] = None
    threshold: float = 0.5
    hidden_dim: int = 64
    num_layers: int = 3
    use_pretrained: bool = True


@dataclass
class IntentVerifierConfig:
    """Configuration for intent verification module."""
    enabled: bool = True
    model_name: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.75
    cache_embeddings: bool = True
    max_cache_size: int = 1000


@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""
    enabled: bool = True
    use_docker: bool = False  # Use in-process sandbox by default
    timeout_seconds: float = 30.0
    max_memory_mb: int = 512
    network_disabled: bool = True
    rollback_on_error: bool = True


@dataclass 
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    log_file: Optional[Path] = None
    rotation: str = "10 MB"


@dataclass
class ShieldAgentConfig:
    """Main configuration for ShieldAgent."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    sanitizer: SanitizerConfig = field(default_factory=SanitizerConfig)
    anomaly_detector: AnomalyDetectorConfig = field(default_factory=AnomalyDetectorConfig)
    intent_verifier: IntentVerifierConfig = field(default_factory=IntentVerifierConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # General settings
    fail_closed: bool = True  # Block execution if any defense fails
    collect_metrics: bool = True
    max_tool_rounds: int = 5
    
    @classmethod
    def from_env(cls) -> "ShieldAgentConfig":
        """Load configuration from environment variables."""
        load_dotenv()
        
        # Get provider and model from environment
        provider = os.getenv("LLM_PROVIDER", "openrouter")
        model = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-70b-instruct")
        
        # Load provider-specific API key
        provider_key_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "google": "GOOGLE_AI_API_KEY",
            "google_ai": "GOOGLE_AI_API_KEY",
            "gemini": "GOOGLE_AI_API_KEY",
        }
        api_key = os.getenv(provider_key_map.get(provider.lower(), "OPENROUTER_API_KEY"))
        
        return cls(
            llm=LLMConfig(
                provider=provider,
                model=model,
                api_key=api_key,
            ),
            sanitizer=SanitizerConfig(
                enabled=os.getenv("SANITIZER_ENABLED", "true").lower() == "true",
            ),
            anomaly_detector=AnomalyDetectorConfig(
                enabled=os.getenv("ANOMALY_DETECTION_ENABLED", "true").lower() == "true",
                threshold=float(os.getenv("ANOMALY_SCORE_THRESHOLD", "0.5")),
                model_path=Path(os.getenv("ANOMALY_MODEL_PATH")) if os.getenv("ANOMALY_MODEL_PATH") else None,
            ),
            intent_verifier=IntentVerifierConfig(
                enabled=os.getenv("INTENT_VERIFICATION_ENABLED", "true").lower() == "true",
                similarity_threshold=float(os.getenv("INTENT_SIMILARITY_THRESHOLD", "0.75")),
            ),
            sandbox=SandboxConfig(
                enabled=os.getenv("SANDBOX_ENABLED", "true").lower() == "true",
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
            ),
        )
    
    @classmethod
    def load_yaml(cls, path: Path) -> "ShieldAgentConfig":
        """Load configuration from YAML file."""
        import yaml
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            llm=LLMConfig(**data.get("llm", {})),
            sanitizer=SanitizerConfig(**data.get("sanitizer", {})),
            anomaly_detector=AnomalyDetectorConfig(**data.get("anomaly_detector", {})),
            intent_verifier=IntentVerifierConfig(**data.get("intent_verifier", {})),
            sandbox=SandboxConfig(**data.get("sandbox", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            fail_closed=data.get("fail_closed", True),
            collect_metrics=data.get("collect_metrics", True),
        )
