# Audio Summarization and Question-Answering System

## Overview

This project is a background system software designed to capture audio conversations and utilize an AI model to summarize and provide answers based on the captured dialogue. The system aims to streamline information retrieval from conversations, making it easier to reference important points or seek clarifications.

## Features

- **Background Audio Capture**: The software runs quietly in the background, recording audio without interrupting the user.
- **AI-Driven Summarization**: Utilizes advanced AI models to process the recorded audio and generate concise summaries.
- **Interactive Q&A**: Users can ask questions based on the conversation, and the system provides relevant answers derived from the summarized content.

## Requirements

1. **Voicemeeter**: A virtual audio mixer required for routing audio input to the application.
2. **Python 3.x**: Ensure you have Python installed to run the application.

## Setup Instructions

1. **Download and Install Voicemeeter**:
   - Visit the [Voicemeeter website](https://vb-audio.com/Voicemeeter/) to download the software.
   - Follow the installation instructions provided on the website.

2. **Configure Voicemeeter**:
   - Open Voicemeeter and set the **A1 routing** to your desired audio output (e.g., speakers or headphones).

3. **Clone or Download the Project**:
   - Clone this repository using `git clone <repository-url>` or download the ZIP file and extract it.

4. **Install Required Dependencies**:
   - Open a terminal in the project directory and run:
     ```bash
     pip install -r requirements.txt
     ```

5. **Run the Application**:
   - Execute the application by running the compiled `.exe` file located in the project folder.

## Usage

Once the application is running, it will start capturing audio. You can interact with the AI model by asking questions or requesting summaries. The system will process the audio and provide responses based on the captured conversation.

## Troubleshooting

- If you encounter issues with audio capture, ensure that Voicemeeter is configured correctly and that the audio output is set to the correct device.
- If any necessary files (e.g., transcription, chunks, FAISS index) do not exist, the application will create them automatically when it runs for the first time.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Voicemeeter](https://vb-audio.com/Voicemeeter/) for providing an essential tool for audio routing.
- [OpenAI](https://openai.com/) for the AI models that power the summarization and question-answering features.

