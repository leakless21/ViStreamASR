# Gap Analysis

This file tracks bugs and missing features. Each entry should have a corresponding unit test to isolate the behavior and prevent regressions.

## Resolved Bugs

### [BUG-001] AttributeError: 'Namespace' object has no attribute 'model_chunk_size_ms'

**Description:**
When running the application, a crash occurred with the error: `AttributeError: 'Namespace' object has no attribute 'model_chunk_size_ms'`. This happened during the configuration loading phase, specifically when `pydantic-settings` attempted to access settings for the first time within the `initialize_logging` function.

**Root Cause:**
The application uses `argparse` in `src/vistreamasr/cli.py` to define and parse command-line arguments (e.g., `--model.chunk_size_ms`). Concurrently, `pydantic-settings` in `src/vistreamasr/config.py` was also attempting to parse command-line arguments automatically by default. This dual parsing led to a conflict. `argparse` created a `Namespace` object with attributes like `model_chunk_size_ms`, but `pydantic-settings` expected a different format or a different set of attributes, leading to the `AttributeError`.

**Resolution:**
The automatic CLI parsing feature of `pydantic-settings` was disabled. This was achieved by adding `cli_parse_args=False` to the `SettingsConfigDict` in the `ViStreamASRSettings` class in `src/vistreamasr/config.py`. This change ensures that `pydantic-settings` only loads configuration from the TOML file and environment variables, while `argparse` remains solely responsible for handling command-line arguments, thus resolving the conflict.

**Files Modified:**

- `src/vistreamasr/config.py`: Added `cli_parse_args=False` to `ViStreamASRSettings.model_config`.

**Unit Test:**
_(A unit test should be created here to verify that settings load correctly when CLI arguments are present and that no AttributeError is raised.)_

---

## Missing Features

_(This section will be populated as new features are identified.)_
