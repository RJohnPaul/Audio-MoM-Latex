# AudioMoM - Advanced Audio Transcription & Meeting Minutes Generator

![GitHub](https://img.shields.io/github/license/RJohnPaul/Audio-MoM-Latex)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/flask-2.0%2B-green)

An innovative web application that combines state-of-the-art audio transcription technology with AI-powered meeting minutes generation in LaTeX format. This versatile tool helps transform any audio recording into professionally formatted meeting documentation with minimal effort.

## 🌟 Features

### Audio Transcription Engine

- **Multi-Model Support**: Utilizes both faster-whisper and WhisperX for accurate transcription
- **Speaker Identification**: Optional diarization to identify different speakers
- **Customizable Processing**: Choose from various model sizes to balance speed and accuracy
- **Language Detection**: Automatic language identification or manual selection
- **Timestamp Integration**: Option to include precise timestamps in output

### Dynamic File Management

- **Drag & Drop Interface**: Modern, intuitive file upload experience
- **Wide Format Support**: Compatible with MP3, WAV, M4A, OGG, FLAC, and AAC files
- **Large File Handling**: Process files up to 100MB
- **Real-time Progress**: Visual feedback during processing

### AI-Powered Minutes of Meeting Generation

- **Smart Text Analysis**: Automatically extracts key topics and decision points
- **LaTeX Document Creation**: Generates professionally formatted documents
- **Topic Clustering**: Groups related discussions for better organization
- **Action Item Detection**: Identifies and highlights action items
- **Export Options**: Download as LaTeX or copy to clipboard

### Sleek Modern Interface

- **Responsive Design**: Works on all devices from desktop to mobile
- **Dark Mode Support**: Comfortable viewing in any environment
- **Intuitive Navigation**: Easy switching between transcription and MoM generation
- **Interactive Previews**: Real-time word count and formatting feedback

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RJohnPaul/Audio-MoM-Latex.git
   cd Audio-MoM-Latex
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create a .env file with your API keys
   cp .env.example .env
   # Edit the .env file with your actual API keys
   ```

### Running the Application

1. Start the Flask server:
   ```bash
   python main.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8080
   ```

## 🔧 Configuration Options

### Transcription Settings

| Setting | Options | Description |
|---------|---------|-------------|
| Model Size | tiny, base, small, medium | Larger models provide better accuracy but slower processing |
| Language | Auto-detect or specific language | Force a specific language or let the model detect it |
| Timestamps | Enable/Disable | Include timing information in the transcript |
| Speaker ID | Enable/Disable | Identify different speakers in the recording |

### MoM Generation Settings

The application intelligently processes transcripts using:
- Semantic clustering to group related topics
- Natural language processing to extract key points
- AI summarization to condense discussions
- LaTeX formatting for professional output

## 🗂️ Project Structure

```
.
├── audio/                  # Sample audio files and transcriptions
├── templates/              # HTML templates for the web interface
│   ├── index.html          # Main application template
│   └── ...                 # Additional UI components
├── whisperx/               # WhisperX integration for advanced transcription
│   ├── __init__.py
│   ├── alignment.py
│   ├── asr.py
│   ├── diarize.py          # Speaker identification functionality
│   ├── transcribe.py
│   ├── whisperx_api.py     # API wrapper for WhisperX
│   └── ...
├── main.py                 # Flask application and main entry point
├── requirements.txt        # Project dependencies
├── .env.example            # Template for environment variables
└── README.md               # This documentation file
```

## 💡 Use Cases

- **Business Meetings**: Automatically document and format meeting discussions
- **Academic Lectures**: Transcribe educational content with precise formatting
- **Interviews**: Convert interview recordings to searchable text
- **Conference Calls**: Generate professional minutes from virtual meetings
- **Research**: Transcribe and organize qualitative research interviews

## 🧠 Technical Implementation

### Backend (Flask)

- RESTful API endpoints for transcription and document generation
- Modular architecture for easy extension and maintenance
- Efficient handling of concurrent requests
- Comprehensive error handling and validation

### Audio Processing

- Integration with state-of-the-art speech recognition models
- Optimized audio preprocessing for improved recognition
- Support for speaker diarization through PyAnnote
- Flexible output formatting options

### AI Document Generation

- Semantic clustering using similarity metrics
- Multi-stage summarization for balanced detail
- Context-aware LaTeX formatting
- Action item and decision point extraction

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Base transcription technology
- [WhisperX](https://github.com/m-bain/whisperX) - Enhanced transcription with word-level timestamps
- [PyAnnote](https://github.com/pyannote/pyannote-audio) - Speaker diarization capability
- [Groq](https://groq.com) - AI inference for document generation
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Alpine.js](https://alpinejs.dev/) - JavaScript framework for UI reactivity
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework

---

<p align="center">
  <i>Frontend Created with ❤️ by <a href="https://github.com/RJohnPaul">RJohnPaul</a> , Backend by Sai Ganesh and Shubam </i>
</p>
