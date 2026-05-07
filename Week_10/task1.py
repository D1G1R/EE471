from transformers import pipeline, logging
from PIL import Image
import requests
import torch
import warnings

# Uyarı mesajlarını temizlemek için
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# 1. Sentiment Analysis
sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentences = [
    "I've been waiting for a EE471 course my whole life.",
    "I hate EE471 course"
]

for s in sentences:
    result = sentiment_pipe(s)
    print(f"'{s}' -> {result}")

# 2. Zero-shot Classification
zero_shot_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zs_text = "Berkshire keeps their cash reserves at an extremely high level."
candidate_labels = ["finance", "sports", "politics"]
zs_result = zero_shot_pipe(zs_text, candidate_labels)
print(f"Text: {zs_text}\nLabels: {zs_result['labels']}\nScores: {zs_result['scores']}")

# 3. Text Generation (Kısa çıktı için max_new_tokens eklendi)
generator = pipeline("text-generation", model="gpt2")
prompt = "If I continue to successfully complete all in-class exercises in EE471 course,"
gen_results = generator(prompt, max_new_tokens=35, num_return_sequences=2, truncation=True, pad_token_id=50256)

print(f"\n{'='*20} TEXT GENERATION RESULTS {'='*20}\n")
print(f"Prompt: {prompt}\n")

for i, res in enumerate(gen_results):
    print(f"[Alternative {i+1}]")
    print(f"{res['generated_text']}")
    print(f"\n{'-'*65}\n")

# 4. Mask Filling
fill_mask = pipeline("fill-mask", model="distilroberta-base")
mask_text = f"To understand generative AI, one must study {fill_mask.tokenizer.mask_token} well."
mask_result = fill_mask(mask_text)
for res in mask_result[:2]:
    print(f"Filled: {res['sequence']} (Score: {res['score']:.4f})")

ner = pipeline("ner", aggregation_strategy="simple")
ner("I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye.")


# 5. Named Entity Recognition (NER)
ner_pipe = pipeline("ner", model="sartajbhuvaji/bert-named-entity-recognition", aggregation_strategy="simple")
ner_text = "I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye."
ner_results = ner_pipe(ner_text)
for entity in ner_results:
    print(f"Entity: {entity['word']} | Label: {entity['entity_group']}")

# 6. Question Answering
ner_text = "I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye."

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

qa_model_name = "deepset/roberta-base-squad2"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

questions = [
    "What is the name of the subject?",
    "Which organization do they work for?",
    "Where do they live?"
]

for q in questions:
    inputs = qa_tokenizer(q, ner_text, return_tensors="pt")
    with torch.no_grad():
        outputs = qa_model(**inputs)
    
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax()
    answer_tokens = inputs.input_ids[0, answer_start : answer_end + 1]
    ans = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    print(f"Q: {q}\nA: {ans}")

# 7. Summarize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/bart-large-cnn" # "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text = """
        The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century,
often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States
housing bubble, its effects rippled across the globe, leading to the collapse of major financial
institutions and a deep international recession. The crisis began with the subprime mortgage market.
In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk loans
to borrowers with poor credit.
"""

inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True).input_ids
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(summary)

# 8. Translate
from transformers import MarianTokenizer, MarianMTModel

model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression."

inputs = tokenizer(text, return_tensors="pt", padding=True)
translated = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)

print(tokenizer.decode(translated[0], skip_special_tokens=True))

# 9. Image Classification
from PIL import Image
import requests
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(img_url, stream=True).raw)
img_result = image_classifier(image)
print(f"Top prediction: {img_result[0]['label']} (Score: {img_result[0]['score']:.4f})")

# 10. Automatic Speech Recognition (Hata düzeltildi)
from transformers import pipeline
import torch

# Adding chunk_length_s avoids the 'num_frames' error by processing the stream in segments
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cuda" if torch.cuda.is_available() else "cpu", chunk_length_s=30)
audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
try:
    asr_result = asr_pipe(audio_url)
    print(f"Transcription: {asr_result['text']}")
except Exception as e:
    print(f"Audio processing failed: {e}")