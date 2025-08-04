# ViStreamASR Developer Guide

## 1. Introduction

Welcome to the developer guide for ViStreamASR. This document provides a comprehensive overview of the project, its architecture, and the development workflow. It is intended for developers who want to contribute to the project or integrate ViStreamASR into their own applications.

## 2. Getting Started

### 2.1. Environment Setup

The project uses `pixi` for dependency management. To set up the development environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/leakless21/ViStreamASR.git
cd ViStreamASR

# Install dependencies
pixi install

# Activate the environment
pixi shell
```

Alternatively, you can use `uv`:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .[dev]
```

### 2.2. Running the Application

Once the environment is set up, you can run the CLI:

```bash
# Transcribe a file
vistream-asr transcribe path/to/your/audio.wav

# Start microphone transcription
vistream-asr microphone
```

## 3. Project Architecture

ViStreamASR is built on a modular, domain-driven architecture. For a complete overview of the system's design, data flow, and component interactions, please refer to the main architecture document:

- **[📖 Architecture Document](ARCHITECTURE.md)**

## 4. Component Deep Dive

Each component of the ViStreamASR system is documented in detail. These documents provide information about the component's responsibilities, API, and implementation.

- **[Component Overview](COMPONENT_OVERVIEW.md)**: A high-level look at all the components and their relationships.
- **[CLI Interface](COMPONENT_CLI_INTERFACE_DOCS.md)**: The command-line interface.
- **[Configuration System](COMPONENT_CONFIGURATION_DOCS.md)**: The hierarchical configuration system.
- **[Core Processing Engine](COMPONENT_CORE_PROCESSING_DOCS.md)**: The core ASR engine.
- **[Logging System](COMPONENT_LOGGING_DOCS.md)**: The structured logging system.
- **[Streaming Interface](COMPONENT_STREAMING_INTERFACE_DOCS.md)**: The streaming API.
- **[VAD Integration](COMPONENT_VAD_INTEGRATION_DOCS.md)**: The voice activity detection component.

## 5. Development Workflow

### 5.1. Testing

The project uses `pytest` for testing. To run the tests:

```bash
pixi run test
```

To run tests with coverage:

```bash
pixi run pytest --cov=vistreamasr --cov-report=html
```

### 5.2. Code Style and Linting

We use `black`, `isort`, and `flake8` to maintain a consistent code style. To format and lint the code:

```bash
# Format the code
pixi run format

# Lint the code
pixi run lint
```

### 5.3. Building the Package

To build the source and wheel distributions:

```bash
pixi run build
```

The built packages will be in the `dist/` directory.

## 6. Contribution Guidelines

We welcome contributions to ViStreamASR. Please see our contribution guidelines for more information on how to get involved.

- **[🤝 Contribution Guidelines](CONTRIBUTING.md)**