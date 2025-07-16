#!/usr/bin/env python3
"""
Simple transcription script that avoids TensorFlow dependencies
"""
import sys
import os

def simple_transcribe(audio_path, model_name="base"):
    try:
        # Import only what we need for basic transcription
        from faster_whisper import WhisperModel
        
        print(f"Loading Whisper model: {model_name}")
        model = WhisperModel(model_name, device="cpu", compute_type="float32")
        
        print(f"Transcribing: {audio_path}")
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Write results
        base_name = os.path.splitext(audio_path)[0]
        
        with open(f"{base_name}_simple.txt", "w", encoding="utf-8") as f:
            for segment in segments:
                text = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                print(text)
                f.write(text + "\n")
        
        print(f"Transcription saved to {base_name}_simple.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_transcribe.py <audio_file> [model_name]")
        print("Example: python simple_transcribe.py audio/audio2.mp3 base")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)
    
    success = simple_transcribe(audio_file, model_name)
    sys.exit(0 if success else 1)
