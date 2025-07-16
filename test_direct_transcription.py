#!/usr/bin/env python3
"""
Test script to directly test the transcription endpoint
"""

import requests
import os

def test_transcription_with_file():
    """Test transcription with the existing audio file"""
    audio_file_path = "audio/audio2.mp3"
    
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return
    
    print(f"Testing transcription with file: {audio_file_path}")
    
    # Prepare the request
    url = "http://localhost:8080/transcribe-audio"
    
    with open(audio_file_path, 'rb') as audio_file:
        files = {
            'audio_file': ('audio2.mp3', audio_file, 'audio/mpeg')
        }
        
        data = {
            'language': 'auto',
            'include_timestamps': 'true',
            'speaker_identification': 'false',
            'model': 'base'
        }
        
        print("Sending request to server...")
        
        try:
            response = requests.post(url, files=files, data=data, timeout=300)
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("SUCCESS!")
                print(f"Transcription length: {len(result.get('transcription', ''))}")
                print(f"Language: {result.get('language')}")
                print(f"Language probability: {result.get('language_probability')}")
                print("First 200 characters of transcription:")
                print(result.get('transcription', '')[:200] + "...")
            else:
                print("ERROR!")
                print(f"Response text: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_transcription_with_file()
