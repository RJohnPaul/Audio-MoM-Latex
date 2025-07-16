#!/usr/bin/env python3
"""
Test script to verify the audio transcription functionality
"""

import requests
import os

def test_transcription_endpoint():
    """Test the transcription endpoint with a simple audio file"""
    # Test if the endpoint is accessible
    try:
        response = requests.get('http://localhost:8080/audio-transcription')
        if response.status_code == 200:
            print("✓ Audio transcription page is accessible")
        else:
            print(f"✗ Audio transcription page returned status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to server: {e}")
        print("Make sure the Flask app is running with 'python main.py'")

def test_file_upload():
    """Test file upload capabilities"""
    # Check if there are any audio files in the audio directory
    audio_dir = "audio"
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'))]
        if audio_files:
            print(f"✓ Found {len(audio_files)} audio file(s) in audio directory:")
            for file in audio_files[:3]:  # Show first 3 files
                print(f"  - {file}")
        else:
            print("⚠ No audio files found in audio directory")
    else:
        print("⚠ Audio directory does not exist")

if __name__ == "__main__":
    print("Testing Audio Transcription Functionality")
    print("=" * 50)
    
    test_transcription_endpoint()
    test_file_upload()
    
    print("\nTo test the full functionality:")
    print("1. Start the Flask app: python main.py")
    print("2. Open http://localhost:8080 in your browser")
    print("3. Click on 'Transcribe' in the sidebar")
    print("4. Upload an audio file and test transcription")
