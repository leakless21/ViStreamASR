# ViStreamASR: Real-Time Streaming ASR Engine

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

ViStreamASR is a high-performance, real-time streaming speech-to-text engine designed for low-latency transcription. It is built with Python and leverages powerful libraries to provide a robust and extensible ASR solution.

![ViStreamASR Demo](resource/u2.gif)

## Features

- **Low-Latency Streaming:** Provides real-time transcription with minimal delay, suitable for live applications.
- **VAD Integration:** Uses Voice Activity Detection (VAD) to intelligently segment speech, improving accuracy and reducing processing overhead.
- **Web Interface:** Includes a user-friendly web interface for live transcription and demonstration.
- **Extensible Architecture:** Designed with a modular architecture that allows for future expansion, including support for different ASR backends and speaker diarization.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+**
- **FFmpeg:** Required for audio capture and processing. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html) or install it via your system's package manager.
  - **On macOS:** `brew install ffmpeg`
  - **On Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`
  - **On Windows:** Download the binaries and add the `bin` directory to your system's PATH.
- **Pixi:** This project uses [Pixi](https://pixi.sh/) for environment and dependency management.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/ViStreamASR.git
    cd ViStreamASR
    ```

2.  **Install dependencies using Pixi:**
    This command will create a virtual environment and install all the necessary dependencies specified in the `pixi.toml` file.
    ```bash
    pixi install
    ```

## Usage

ViStreamASR can be run in two modes: via the command-line interface (CLI) or as a web server with a graphical interface.

### 1. Command-Line Interface (CLI)

The CLI is ideal for quick tests or for integrating the ASR engine into scripted workflows. It will transcribe audio from your default microphone in real-time.

To start the CLI, run the following command from the project root:

```bash
pixi run start-cli
```

You will see the transcribed text printed to your console as you speak.

### 2. Web Interface

The web interface provides a user-friendly way to interact with the ASR engine. It displays the transcribed text in real-time in your browser.

To start the web server, run the following command:

```bash
pixi run start-web
```

Once the server is running, open your web browser and navigate to:

[http://localhost:8000](http://localhost:8000)

You will see the live transcription in the browser window.

## How It Works

For a detailed explanation of the project's architecture, components, and future plans, please refer to the documents in the [`docs/`](docs/) directory, including:

- **[`ARCHITECTURE.md`](docs/ARCHITECTURE.md):** A deep dive into the system's design.
- **[`REQUIREMENTS.md`](docs/REQUIREMENTS.md):** A full list of functional and non-functional requirements.
- **[`COMPONENT_OVERVIEW.md`](docs/COMPONENT_OVERVIEW.md):** An overview of the different components and their roles.

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.
