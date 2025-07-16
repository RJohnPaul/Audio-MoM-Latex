meeting_transcript='''So, just to kick things off, I wanted to mention that the UI overhaul is finally complete on staging. We’re now just waiting for feedback from the design team before pushing it to production.

Right, and we also need to double-check how the new color scheme looks on smaller screens. Someone mentioned it was a bit off on mobile last time.

Yeah, I noticed that too—especially on iPhone SE and similar compact viewports. We might need to tweak the padding a bit on those cards.

That reminds me, what's the timeline for the analytics dashboard? Weren’t we aiming to get it done by next Friday?

We were, but there’s been a slight delay due to the API integration. The data’s not coming through as expected from the CRM endpoint.

We could mock the data for now, just to unblock the frontend work. Then plug in the actual feed once it's fixed.

That’s a good idea. Also, about the feedback we got on the onboarding flow—should we start implementing those suggestions now or wait for the next sprint?

Honestly, some of them are pretty quick fixes. Like changing the tooltip text and swapping a few icons. I think we can slide them into this sprint without too much disruption.

Agreed. The bigger requests—like the multi-step walkthrough—can be scoped for the next cycle. But let’s patch what we can this week.

Before we wrap, one more thing: the export-to-PDF feature. Who’s taking that on?

I think it was unassigned. But it makes sense to pair it with the report summary task—there’s overlap.

Perfect. Let’s make sure we add it to the board. Alright, anything else before we close?

Nope, that’s all from my side.

Sounds good—thanks, everyone.'''
from flask import Flask, render_template, request, jsonify
import re
import numpy as np
from numpy.linalg import norm
from numpy import dot
from groq import Groq
import os
import tempfile
from werkzeug.utils import secure_filename
import faster_whisper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

client = Groq(api_key="KEY_PLACEHOLDER")  # Replace with your actual Groq API key

def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def chunk_text(text, max_words=150):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def cluster_meeting_transcript(transcript):
    sentences = re.split(r'(?<=[.!?]) +', transcript)
    chunk_size = 5
    step_size = 2
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), step_size)]

    embeddings = [np.random.rand(384) for _ in chunks]
    similarities = [
        dot(embeddings[i], embeddings[i + 1]) / (norm(embeddings[i]) * norm(embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    smoothed = []
    window = 3
    for i in range(len(similarities)):
        start, end = max(0, i - window // 2), min(len(similarities), i + window // 2 + 1)
        smoothed.append(np.mean(similarities[start:end]))

    avg, std = np.mean(smoothed), np.std(smoothed)
    threshold = avg - 1.2 * std
    boundaries = [0] + [i + 1 for i, s in enumerate(smoothed) if s < threshold]

    clusters = []
    for i in range(1, len(boundaries)):
        clusters.append(' '.join(chunks[boundaries[i - 1]:boundaries[i]]))
    if boundaries[-1] < len(chunks):
        clusters.append(' '.join(chunks[boundaries[-1]:]))

    return clusters

def generate_summary(text, mode="detailed"):
    if mode == "detailed":
        prompt = f"""
Please summarize the following segment of a meeting transcript in a clear and detailed way.
Capture all relevant points and insights from the text only:\n\n{text}
"""
    else:
        prompt = f"""
Summarize the following meeting transcript in a concise, high-level way.
Extract only the key topics and insights based on the content:\n\n{text}
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        return remove_think_tags(chat_completion.choices[0].message.content)
    except Exception as e:
        return f"Error contacting Groq API: {e}"

def generate_latex_mom(summary_text):
    prompt = f"""
You are a LaTeX expert. Create a fully compilable LaTeX document for "Minutes of Meeting" (MoM) based ONLY on the content below. Do NOT include empty sections. Do NOT fabricate any information.

# Instructions:
- Output ONLY LaTeX code.
- Use clean, professional formatting for business MoMs.
- Include sections ONLY if relevant and present in the summary.
- The document must compile successfully in Overleaf without errors.

# Summary:
{summary_text}
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        return remove_think_tags(chat_completion.choices[0].message.content)
    except Exception as e:
        return f"Error generating LaTeX: {e}"

ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def transcribe_audio_file(file_path, model_size="base", language=None, include_timestamps=True, speaker_identification=False):
    """
    Transcribe audio file using faster-whisper and optionally WhisperX for speaker identification
    """
    try:
        if speaker_identification:
            # Use WhisperX for speaker identification
            from whisperx.whisperx_api import WhisperXTranscriber
            
            transcriber = WhisperXTranscriber(
                model_name=model_size,
                device="cpu",
                compute_type="float32"
            )
            
            vad_options = {
                "vad_method": "pyannote",
                "vad_onset": 0.500,
                "vad_offset": 0.363,
                "chunk_size": 30
            }
            
            result = transcriber.transcribe(
                audio_path=file_path,
                language=language if language != 'auto' else 'en',
                diarize=True,
                align=True,
                vad_options=vad_options,
                output_dir="temp",
                output_format="none"
            )
            
            if result and len(result) > 0:
                first_result = result[0]["result"]
                
                # Format the transcription with speaker identification
                transcription_text = ""
                for segment in first_result["segments"]:
                    speaker = segment.get("speaker", "Unknown")
                    text = segment["text"].strip()
                    if include_timestamps:
                        start_time = segment.get("start", 0)
                        end_time = segment.get("end", 0)
                        transcription_text += f"[{start_time:.2f}s -> {end_time:.2f}s] [{speaker}]: {text}\n"
                    else:
                        transcription_text += f"[{speaker}]: {text}\n"
                
                transcriber.cleanup()
                
                return {
                    'success': True,
                    'transcription': transcription_text,
                    'language': first_result.get("language", "en"),
                    'language_probability': 0.95
                }
            else:
                transcriber.cleanup()
                return {
                    'success': False,
                    'error': 'No transcription result returned'
                }
        else:
            # Use faster-whisper for basic transcription
            model = faster_whisper.WhisperModel(model_size, device="cpu", compute_type="float32")
            
            # Transcribe the audio
            segments, info = model.transcribe(
                file_path, 
                beam_size=5,
                language=language if language != 'auto' else None,
                word_timestamps=include_timestamps
            )
            
            # Format the transcription
            transcription_text = ""
            for segment in segments:
                if include_timestamps:
                    transcription_text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                else:
                    transcription_text += f"{segment.text}\n"
            
            return {
                'success': True,
                'transcription': transcription_text,
                'language': info.language,
                'language_probability': info.language_probability
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route("/", methods=["GET", "POST"])
def index():
    latex_code = ""
    transcript = ""

    if request.method == "POST":
        transcript = request.form.get("transcript", "")
        if transcript.strip():
            clusters = cluster_meeting_transcript(transcript)
            cluster_summaries = []

            for cluster in clusters:
                if len(cluster.split()) > 300:
                    chunks = chunk_text(cluster)
                    summaries = [generate_summary(c) for c in chunks]
                    cluster_summaries.append(" ".join(summaries))
                else:
                    cluster_summaries.append(generate_summary(cluster))

            combined_summary = " ".join(cluster_summaries)
            final_summary = generate_summary(combined_summary, mode="highlevel")
            latex_code = generate_latex_mom(final_summary)

    return render_template("index.html", latex_code=latex_code, transcript=transcript)

@app.route("/generate-mom", methods=["POST"])
def generate_mom():
    try:
        transcript = request.form.get("transcript", "")
        if not transcript.strip():
            return jsonify({'error': 'No transcript provided'}), 400
            
        clusters = cluster_meeting_transcript(transcript)
        cluster_summaries = []

        for cluster in clusters:
            if len(cluster.split()) > 300:
                chunks = chunk_text(cluster)
                summaries = [generate_summary(c) for c in chunks]
                cluster_summaries.append(" ".join(summaries))
            else:
                cluster_summaries.append(generate_summary(cluster))

        combined_summary = " ".join(cluster_summaries)
        final_summary = generate_summary(combined_summary, mode="highlevel")
        latex_code = generate_latex_mom(final_summary)
        
        return jsonify({'latex_code': latex_code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio():
    try:
        print(f"Received transcribe-audio request")
        print(f"Files in request: {list(request.files.keys())}")
        print(f"Form data: {dict(request.form)}")
          # Check if file was uploaded
        if 'audio_file' not in request.files:
            print("Error: No audio file provided")
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio_file']
        print(f"File received: {file.filename}")
        print(f"File content type: {file.content_type}")
        print(f"File stream position: {file.stream.tell()}")
        
        # Reset stream position to beginning
        file.stream.seek(0)
        
        if file.filename == '':
            print("Error: No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print(f"Error: Invalid file type for {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload MP3, WAV, M4A, OGG, FLAC, or AAC files.'}), 400
          
        # Get settings from form
        language = request.form.get('language', 'auto')
        include_timestamps = request.form.get('include_timestamps', 'true').lower() == 'true'
        speaker_identification = request.form.get('speaker_identification', 'false').lower() == 'true'
        model_size = request.form.get('model', 'base')
        
        print(f"Settings: language={language}, timestamps={include_timestamps}, speaker_id={speaker_identification}, model={model_size}")
          # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        print(f"Saving file to: {temp_path}")
        file.save(temp_path)
        
        # Verify file was saved correctly
        if not os.path.exists(temp_path):
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        file_size = os.path.getsize(temp_path)
        print(f"File saved successfully, size: {file_size} bytes")
        
        if file_size == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        try:            
            print("Starting transcription...")
            # Transcribe the audio
            result = transcribe_audio_file(
                temp_path, 
                model_size=model_size,
                language=language,
                include_timestamps=include_timestamps,
                speaker_identification=speaker_identification
            )
            
            print(f"Transcription result: success={result.get('success')}")
            if result.get('success'):
                transcription_length = len(result.get('transcription', ''))
                print(f"Transcription length: {transcription_length} characters")
                
                return jsonify({
                    'transcription': result['transcription'],
                    'language': result['language'],
                    'language_probability': result['language_probability']
                })
            else:
                print(f"Transcription failed: {result.get('error')}")
                return jsonify({'error': result['error']}), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
                
    except Exception as e:
        print(f"Exception in transcribe_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route("/audio-transcription")
def audio_transcription():
    """Route specifically for audio transcription page"""
    return render_template("index.html")

@app.route("/test-upload")
def test_upload():
    """Test upload page"""
    return app.send_static_file('../test_upload.html')

@app.route("/debug-upload")
def debug_upload():
    """Debug upload page"""
    return app.send_static_file('../debug_upload.html')

@app.route('/debug')
def debug_test():
    with open('debug_test.html', 'r') as f:
        return f.read()
    
@app.route('/minimal')
def minimal_test():
    with open('minimal_test.html', 'r') as f:
        return f.read()

@app.route('/simple')
def simple_test():
    with open('simple_test.html', 'r') as f:
        return f.read()

if __name__ == "__main__":
    app.run(debug=True, port=8080, use_reloader=False)
