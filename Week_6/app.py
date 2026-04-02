
import os
from flask import Flask, render_template, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import time

# .env dosyasındaki değişkenleri yükle
load_dotenv()

app = Flask(__name__)

# Static klasörünün var olduğundan emin ol
os.makedirs('static', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Lütfen bir metin girin.'}), 400

    try:
        # Eski dosyayı sil
        filename = "static/output.wav"
        if os.path.exists(filename):
            os.remove(filename)

        # Azure kimlik bilgilerini al
        speech_key = os.getenv('SPEECH_KEY')
        speech_region = os.getenv('SPEECH_REGION')

        # Speech konfigürasyonunu ayarla
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        
        # Sesi bir dosyaya kaydetmek için ayar (Arayüzde çalabilmek için)
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)

        # Synthesizer'ı oluştur
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        # Metni sese çevir
        result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return jsonify({'message': 'Audio successfully created!', 'audio_url': f'/{filename}?v={int(time.time())}'})
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            return jsonify({'error': f'Hata: {cancellation_details.reason}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)