# Contributing to ViStreamASR

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create your feature branch (`git checkout -b feature/amazing-feature`)
2. **Follow the code style** guidelines (Black, isort, flake8)
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description of your changes

## Development Guidelines

- Use type hints for all public APIs
- Write comprehensive docstrings following Google style
- Include unit tests for new features
- Update relevant documentation
- Follow existing code patterns and conventions

## VAD Development Notes

When working with VAD integration:

- Test with various Vietnamese speech patterns and dialects
- Ensure VAD parameters work well with default ASR chunk sizes
- Verify performance improvements don't impact accuracy
- Test edge cases (very short speech, background noise, etc.)

## Configuration & Logging Development Notes

When working with the configuration and logging system:

- Ensure configuration validation covers all edge cases
- Test configuration loading from all sources (TOML, env vars, CLI)
- Verify logging works with different output formats and sinks
- Test log rotation and retention policies
- Ensure backward compatibility with legacy parameter passing
