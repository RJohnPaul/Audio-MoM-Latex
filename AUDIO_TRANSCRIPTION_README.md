# Audio Transcription Features

## Overview
The audio transcription functionality has been enhanced to provide a dynamic, user-friendly interface for uploading audio files and generating transcriptions using the Whisper model.

## Features

### 1. Dynamic File Upload
- **Drag & Drop Support**: Users can drag and drop audio files or click to browse
- **File Type Validation**: Supports MP3, WAV, M4A, OGG, FLAC, and AAC files
- **File Size Limit**: Maximum 100MB file size
- **Real-time File Preview**: Shows selected file information

### 2. Transcription Settings
- **Language Detection**: Auto-detect language or manually select from supported languages
- **Model Selection**: Choose from different Whisper model sizes:
  - `tiny`: Fastest processing, lower accuracy
  - `base`: Balanced speed and accuracy (default)
  - `small`: Slower processing, higher accuracy
  - `medium`: Slowest processing, highest accuracy
- **Timestamp Options**: Include or exclude timestamps in output
- **Speaker Identification**: Optional speaker diarization using WhisperX (requires additional processing time)

### 3. Processing & Results
- **Real-time Progress**: Visual progress indicator during transcription
- **Dynamic Status Updates**: Shows processing status based on selected options
- **Formatted Output**: Clean, readable transcription results
- **Word Count**: Displays word count for completed transcriptions

### 4. Integration with MoM Generator
- **One-Click Transfer**: Use transcription results directly in the Minutes of Meeting generator
- **Seamless Workflow**: Automatic navigation and data population

### 5. Export Options
- **Download Transcription**: Save transcription as a text file
- **Filename Generation**: Automatic filename based on original audio file

## Technical Implementation

### Backend (Flask)
- **Endpoint**: `/transcribe-audio` (POST)
- **Models**: Supports both faster-whisper and WhisperX
- **Speaker Identification**: Uses PyAnnote for diarization when enabled
- **Error Handling**: Comprehensive error handling and user feedback

### Frontend (HTML/JavaScript)
- **Framework**: Alpine.js for reactive functionality
- **Styling**: Tailwind CSS for modern UI
- **File Handling**: Native HTML5 file API
- **AJAX**: Fetch API for asynchronous requests

## Usage Instructions

1. **Start the Application**:
   ```bash
   python main.py
   ```

2. **Access the Interface**:
   - Open http://localhost:8080 in your browser
   - Click "Transcribe" in the sidebar

3. **Upload Audio**:
   - Drag and drop an audio file or click "Upload a file"
   - Configure transcription settings as needed
   - Click "Start Transcription"

4. **View Results**:
   - Wait for processing to complete
   - Review the transcription in the results panel
   - Download or use for MoM generation

## File Structure
```
templates/
  └── index.html          # Main UI template with transcription interface
main.py                   # Flask application with transcription endpoints
whisperx/                 # WhisperX integration for speaker identification
  ├── whisperx_api.py     # Main WhisperX API wrapper
  └── ...                 # Supporting modules
test_audio_transcription.py  # Test script for functionality verification
```

## Dependencies
- Flask
- faster-whisper
- whisperx (for speaker identification)
- numpy
- werkzeug

## Notes
- Speaker identification requires additional processing time
- The application uses CPU processing by default
- For production use, consider GPU acceleration for faster processing
- Audio files are temporarily stored during processing and automatically cleaned up
