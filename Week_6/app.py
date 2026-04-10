import os
import shutil
from flask import Flask, render_template, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import time

load_dotenv()
app = Flask(__name__)

STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# --- TEXT TO SPEECH ---
@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'Please enter some text.'}), 400

    try:
        speech_key = os.getenv('SPEECH_KEY')
        speech_region = os.getenv('SPEECH_REGION')
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        
        filename = f"output_{int(time.time())}.wav"
        file_path = os.path.join('/tmp', filename)  # /tmp her zaman yazılabilir
        audio_config = speechsdk.audio.AudioOutputConfig(filename=file_path)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            dest_path = os.path.join(STATIC_DIR, filename)
            shutil.move(file_path, dest_path)
            return jsonify({
                'message': 'Audio successfully created!',
                'audio_url': f'/static/{filename}'
            })
        elif result.reason == speechsdk.ResultReason.Canceled:
            return jsonify({'error': f'Error: {result.cancellation_details.reason}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- SPEECH TO TEXT ---
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'Please upload an audio file.'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(file_path)

    try:
        speech_key = os.getenv('SPEECH_KEY')
        speech_region = os.getenv('SPEECH_REGION')
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        
        audio_config = speechsdk.AudioConfig(filename=file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = speech_recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return jsonify({'text': result.text, 'message': 'Audio successfully transcribed!'})
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return jsonify({'error': 'Speech not recognized. Please upload a clear audio file.'}), 400
        elif result.reason == speechsdk.ResultReason.Canceled:
            return jsonify({'error': 'Operation canceled.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'speech_recognizer' in locals():
            del speech_recognizer
        if 'audio_config' in locals():
            del audio_config
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)