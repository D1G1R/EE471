import os
from flask import Flask, render_template, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'Lütfen bir ses dosyası yükleyin.'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi.'}), 400

    # Dosyayı geçici olarak kaydet
    file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(file_path)

    try:
        speech_key = os.getenv('SPEECH_KEY')
        speech_region = os.getenv('SPEECH_REGION')
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        
        # Sesi dosyadan okuma ayarı
        audio_config = speechsdk.AudioConfig(filename=file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        # Sesi metne çevir (Senkron)
        result = speech_recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return jsonify({'text': result.text, 'message': 'Ses başarıyla metne çevrildi!'})
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return jsonify({'error': 'Ses anlaşılamadı. Lütfen net bir ses dosyası yükleyin.'}), 400
        elif result.reason == speechsdk.ResultReason.Canceled:
            return jsonify({'error': 'İşlem iptal edildi veya Azure bağlantı hatası.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Azure araçlarının dosya üzerindeki kilidini kaldırması için onları bellekten siliyoruz
        if 'speech_recognizer' in locals():
            del speech_recognizer
        if 'audio_config' in locals():
            del audio_config

        # İşlem bitince geçici dosyayı temizle
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)