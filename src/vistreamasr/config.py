"""Configuration management for ViStreamASR."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class ModelConfig(BaseModel):
    """Configuration for the ASR model."""
    
    name: str = Field(
        default="whisper",
        description="Name of the ASR model to use"
    )
    
    chunk_size_ms: int = Field(
        default=640,
        gt=0,
        description="Chunk size in milliseconds for processing audio"
    )
    
    stride_ms: int = Field(
        default=320,
        gt=0,
        description="Stride in milliseconds between chunks"
    )
    
    auto_finalize_after: float = Field(
        default=15.0,
        gt=0.5,
        le=60.0,
        description="Maximum duration in seconds before auto-finalizing a segment"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode for detailed processing information"
    )


class VADConfig(BaseModel):
    """Configuration for Voice Activity Detection."""
    
    enabled: bool = Field(
        default=True,
        description="Enable Voice Activity Detection"
    )
    
    aggressiveness: int = Field(
        default=3,
        ge=0,
        le=3,
        description="VAD aggressiveness level (0-3, 3 being most aggressive)"
    )
    
    frame_size_ms: int = Field(
        default=30,
        gt=0,
        description="Frame size in milliseconds for VAD processing"
    )
    
    min_silence_duration_ms: int = Field(
        default=500,  # Updated from 100 based on WhisperLiveKit research
        gt=0,
        description="Minimum silence duration in milliseconds to mark a segment as final"
    )
    
    speech_pad_ms: int = Field(
        default=30,
        gt=0,
        description="Padding added to speech segments in milliseconds"
    )
    
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate for VAD processing"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging system."""
    
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    rotation: Optional[str] = Field(
        default=None,
        description="Log file rotation condition (e.g., '10 MB', '1 week')"
    )
    
    retention: Optional[str] = Field(
        default=None,
        description="Log file retention period (e.g., '1 week', '6 months')"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to log file. If None, logs to stdout only"
    )
    
    format_string: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        description="Format string for log messages"
    )
    
    enable_colors: bool = Field(
        default=True,
        description="Enable colored output for console logs"
    )


class ViStreamASRSettings(BaseSettings):
    """
    Main settings class for ViStreamASR.
    
    This class provides hierarchical configuration loading with the following priority:
    1. CLI arguments (highest)
    2. Environment variables (with VISTREAMASR_ prefix)
    3. TOML configuration file (lowest)
    """
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    model_config = SettingsConfigDict(
        env_prefix='vistreamasr_',
        env_nested_delimiter='__',
        case_sensitive=False,
        env_file='.env',
        env_file_encoding='utf-8',
        extra='forbid',
        cli_parse_args=False  # Explicitly disable automatic CLI parsing by pydantic-settings
    )
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize settings sources to explicitly exclude CLI parsing.
        This is a more robust way to prevent pydantic-settings from
        interfering with our manual argparse setup.
        """
        # Return all sources except CLI (init_settings typically contains CLI args)
        # The order is important: init_settings > env_settings > dotenv_settings > file_secret_settings
        # Since we want to exclude CLI, we'll return env_settings, dotenv_settings, file_secret_settings
        return (env_settings, dotenv_settings, file_secret_settings)
    @classmethod
    def load_from_toml(cls, config_path: Optional[Path] = None) -> 'ViStreamASRSettings':
        """
        Load settings from a TOML configuration file.
        
        Args:
            config_path: Path to TOML configuration file. If None, looks for
                        'vistreamasr.toml' in current directory and parent directories.
        
        Returns:
            ViStreamASRSettings instance with values loaded from TOML file
        """
        if config_path is None:
            # Look for config file in current and parent directories
            current_dir = Path.cwd()
            for parent in [current_dir] + list(current_dir.parents):
                potential_path = parent / 'vistreamasr.toml'
                if potential_path.exists():
                    config_path = potential_path
                    break
            else:
                # No config file found, return defaults
                return cls()
        
        if not config_path.exists():
            return cls()
        
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError(
                    "No TOML parser available. Please install 'tomli' for Python < 3.11 "
                    "or use Python 3.11+ which includes tomllib."
                )
        
        with open(config_path, 'rb') as f:
            config_data = tomllib.load(f)
        
        # The instantiation of cls() is where pydantic-settings tries to merge
        # TOML, env vars, and CLI args.
        
        settings_instance = cls(**config_data)
        
        return settings_instance
    
    def save_to_toml(self, config_path: Path) -> None:
        """
        Save current settings to a TOML configuration file.
        
        Args:
            config_path: Path where to save the TOML configuration file
        """
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError(
                    "No TOML parser available. Please install 'tomli' for Python < 3.11 "
                    "or use Python 3.11+ which includes tomllib."
                )
        
        try:
            import tomli_w
        except ImportError:
            raise ImportError(
                "No TOML writer available. Please install 'tomli-w'."
            )
        
        # Convert to dict, excluding None values and default values
        config_dict = self.model_dump(exclude_none=True, exclude_defaults=True)
        
        with open(config_path, 'wb') as f:
            tomli_w.dump(config_dict, f)


# Global settings instance
_settings: Optional[ViStreamASRSettings] = None


def get_settings(config_path: Optional[Path] = None) -> ViStreamASRSettings:
    """
    Get the global settings instance, loading from TOML if not already loaded.
    
    Args:
        config_path: Optional path to TOML configuration file
        
    Returns:
        ViStreamASRSettings instance
    """
    global _settings
    if _settings is None:
        # DEBUG: Log sys.argv before settings instantiation
        print(f"DEBUG config.py get_settings: sys.argv = {sys.argv}")
        
        _settings = ViStreamASRSettings.load_from_toml(config_path)
        
        # DEBUG: Confirm model_config after instantiation
        print(f"DEBUG config.py get_settings: ViStreamASRSettings.model_config.cli_parse_args = {_settings.model_config.get('cli_parse_args')}")
        
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
