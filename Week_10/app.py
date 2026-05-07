import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel, AutoModelForQuestionAnswering
from transformers.utils import logging
import torch
from PIL import Image
import warnings

# Uyarıları gizlemek için
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class AITasks:

    def __init__(self):
        # Modelleri önbellekte tutmak için bir sözlük
        self.pipelines = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_sentiment(self, text):
        if 'sentiment' not in self.pipelines:
            self.pipelines['sentiment'] = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=self.device)
        result = self.pipelines['sentiment'](text)[0]
        return f"Label: {result['label']}, Score: {result['score']:.4f}"

    def get_zero_shot(self, text, labels_str):
        if 'zero_shot' not in self.pipelines:
            self.pipelines['zero_shot'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=self.device)
        labels = [label.strip() for label in labels_str.split(",")]
        result = self.pipelines['zero_shot'](text, labels)
        return f"Top Label: {result['labels'][0]}, Score: {result['scores'][0]:.4f}"

    def generate_text(self, prompt):
        if 'text_gen' not in self.pipelines:
            self.pipelines['text_gen'] = pipeline("text-generation", model="gpt2", device=self.device)
        results = self.pipelines['text_gen'](prompt, max_new_tokens=35, num_return_sequences=2, truncation=True, pad_token_id=50256)
        return (f"✨ Alternative 1 ✨\n{results[0]['generated_text']}\n\n"
                f"{'='*40}\n\n"
                f"✨ Alternative 2 ✨\n{results[1]['generated_text']}")

    def fill_mask(self, text):
        if 'fill_mask' not in self.pipelines:
            self.pipelines['fill_mask'] = pipeline("fill-mask", model="distilroberta-base", device=self.device)
        # Eğer kullanıcı mask token eklememişse otomatik ekleyelim
        if "<mask>" not in text:
            text = text.replace("[...]", "<mask>") 
        results = self.pipelines['fill_mask'](text)
        return f"1. {results[0]['sequence']} (Score: {results[0]['score']:.4f})\n2. {results[1]['sequence']} (Score: {results[1]['score']:.4f})"

    def extract_entities(self, text):
        if 'ner' not in self.pipelines:
            self.pipelines['ner'] = pipeline("ner", model="sartajbhuvaji/bert-named-entity-recognition", aggregation_strategy="simple", device=self.device)
        results = self.pipelines['ner'](text)
        entities = [f"{ent['word']} ({ent['entity_group']})" for ent in results]
        return ", ".join(entities) if entities else "No entities found."

    def answer_question(self, context, question):
        if 'qa_tokenizer' not in self.pipelines:
            qa_model_name = "deepset/roberta-base-squad2"
            self.pipelines['qa_tokenizer'] = AutoTokenizer.from_pretrained(qa_model_name)
            self.pipelines['qa_model'] = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(self.device)
            
        inputs = self.pipelines['qa_tokenizer'](question, context, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.pipelines['qa_model'](**inputs)
        
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax()
        answer_tokens = inputs.input_ids[0, answer_start : answer_end + 1]
        return self.pipelines['qa_tokenizer'].decode(answer_tokens, skip_special_tokens=True)

    def summarize(self, text):
        if 'summarize' not in self.pipelines:
            model_id = "facebook/bart-large-cnn"
            self.pipelines['sum_tokenizer'] = AutoTokenizer.from_pretrained(model_id)
            self.pipelines['sum_model'] = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        
        inputs = self.pipelines['sum_tokenizer']("summarize: " + text, return_tensors="pt", truncation=True).input_ids.to(self.device)
        outputs = self.pipelines['sum_model'].generate(inputs, max_new_tokens=100, do_sample=False)
        return self.pipelines['sum_tokenizer'].decode(outputs[0], skip_special_tokens=True)

    def translate(self, text):
        if 'translate' not in self.pipelines:
            model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
            self.pipelines['trans_tokenizer'] = MarianTokenizer.from_pretrained(model_name)
            self.pipelines['trans_model'] = MarianMTModel.from_pretrained(model_name).to(self.device)
        
        inputs = self.pipelines['trans_tokenizer'](text, return_tensors="pt", padding=True).to(self.device)
        translated = self.pipelines['trans_model'].generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
        return self.pipelines['trans_tokenizer'].decode(translated[0], skip_special_tokens=True)

    def classify_image(self, image):
        if 'image_class' not in self.pipelines:
            self.pipelines['image_class'] = pipeline("image-classification", model="google/vit-base-patch16-224", device=self.device)
        # Gradio, numpy array gönderdiği için PIL Image'a çevirmemiz gerekir
        img = Image.fromarray(image)
        result = self.pipelines['image_class'](img)
        return f"Prediction: {result[0]['label']} (Score: {result[0]['score']:.4f})"

    def transcribe_audio(self, audio_path):
        if 'asr' not in self.pipelines:
            self.pipelines['asr'] = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=self.device)
        try:
            result = self.pipelines['asr'](audio_path)
            return result['text']
        except Exception as e:
            return f"Error: {e}"

# Sınıfımızı başlatalım
ai = AITasks()

# Gradio Arayüzünü Oluşturalım
with gr.Blocks(title="EE471 AI Tasks Demo") as demo:
    gr.Markdown("# EE471 Modern Software Development - AI Tasks Demo")
    
    with gr.Tabs():
        with gr.Tab("1. Sentiment Analysis"):
            sa_input = gr.Textbox(label="Input Text", value="I've been waiting for a EE471 course my whole life.")
            sa_btn = gr.Button("Submit")
            sa_output = gr.Textbox(label="Output")
            sa_btn.click(fn=ai.get_sentiment, inputs=sa_input, outputs=sa_output)

        with gr.Tab("2. Zero-shot Classification"):
            zs_input = gr.Textbox(label="Input Text", value="Berkshire keeps their cash reserves at an extremely high level.")
            zs_labels = gr.Textbox(label="Candidate Labels (comma separated)", value="finance, sports, politics")
            zs_btn = gr.Button("Submit")
            zs_output = gr.Textbox(label="Output")
            zs_btn.click(fn=ai.get_zero_shot, inputs=[zs_input, zs_labels], outputs=zs_output)

        with gr.Tab("3. Text Generation"):
            tg_input = gr.Textbox(label="Prompt", value="If I continue to successfully complete all in-class exercises in EE471 course,")
            tg_btn = gr.Button("Submit")
            tg_output = gr.Textbox(label="Generated Text", lines=4)
            tg_btn.click(fn=ai.generate_text, inputs=tg_input, outputs=tg_output)

        with gr.Tab("4. Mask Filling"):
            mf_input = gr.Textbox(label="Text with <mask>", value="To understand generative AI, one must study <mask> well.")
            mf_btn = gr.Button("Submit")
            mf_output = gr.Textbox(label="Output")
            mf_btn.click(fn=ai.fill_mask, inputs=mf_input, outputs=mf_output)

        with gr.Tab("5. Named Entity Recognition"):
            ner_input = gr.Textbox(label="Input Text", value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye.")
            ner_btn = gr.Button("Submit")
            ner_output = gr.Textbox(label="Extracted Entities")
            ner_btn.click(fn=ai.extract_entities, inputs=ner_input, outputs=ner_output)

        with gr.Tab("6. Question Answering"):
            qa_context = gr.Textbox(label="Context", value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye.")
            qa_question = gr.Textbox(label="Question", value="Where do they live?")
            qa_btn = gr.Button("Submit")
            qa_output = gr.Textbox(label="Answer")
            qa_btn.click(fn=ai.answer_question, inputs=[qa_context, qa_question], outputs=qa_output)

        with gr.Tab("7. Summarization"):
            sum_input = gr.Textbox(label="Long Text", lines=5, value="""The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century,
                often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States
                housing bubble, its effects rippled across the globe, leading to the collapse of major financial
                institutions and a deep international recession. The crisis began with the subprime mortgage market.
                In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk loans
                to borrowers with poor credit.""")
            sum_btn = gr.Button("Submit")
            sum_output = gr.Textbox(label="Summary")
            sum_btn.click(fn=ai.summarize, inputs=sum_input, outputs=sum_output)

        with gr.Tab("8. Translation"):
            tr_input = gr.Textbox(label="English Text", value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century.")
            tr_btn = gr.Button("Submit")
            tr_output = gr.Textbox(label="Turkish Translation")
            tr_btn.click(fn=ai.translate, inputs=tr_input, outputs=tr_output)

        with gr.Tab("9. Image Classification"):
            img_input = gr.Image(label="Upload Image")
            img_btn = gr.Button("Submit")
            img_output = gr.Textbox(label="Classification Result")
            img_btn.click(fn=ai.classify_image, inputs=img_input, outputs=img_output)

        with gr.Tab("10. Speech Recognition"):
            audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")
            audio_btn = gr.Button("Submit")
            audio_output = gr.Textbox(label="Transcription")
            audio_btn.click(fn=ai.transcribe_audio, inputs=audio_input, outputs=audio_output)

# Uygulamayı başlat
if __name__ == "__main__":
    demo.launch()