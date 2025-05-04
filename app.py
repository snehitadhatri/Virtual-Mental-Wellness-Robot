from flask import Flask, request, jsonify, send_from_directory
import openai
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os
import json
import datetime
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import cv2
import numpy as np

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure OpenAI API key is set
if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not set. OpenAI API calls will fail.")

# Initialize OpenAI API client
from openai import OpenAI
client = OpenAI(api_key=openai_api_key)

# Flask setup
app = Flask(__name__, static_folder='')

# Load emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

# Load local text generation model and tokenizer for fallback
local_model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(local_model_name)
# Set pad_token to eos_token if pad_token is not set to avoid padding errors
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(local_model_name)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Preprocess text: lowercase, remove stopwords and non-alphabetic characters
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Training data for intent classification
training_texts = [
    "I am frustrated and upset",
    "Can you help me with this problem?",
    "Hello there!",
    "Thank you for your support",
    "I need some advice",
    "Good morning",
    "I am angry and annoyed",
    "Thanks for the help",
    "Hey, how are you?",
    "I appreciate your assistance"
]
training_labels = [
    "venting",
    "asking_for_help",
    "greeting",
    "feedback",
    "asking_for_help",
    "greeting",
    "venting",
    "feedback",
    "greeting",
    "feedback"
]

# Preprocess training texts
training_texts_processed = [preprocess_text(text) for text in training_texts]

# Vectorize texts
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_texts_processed)

# Train logistic regression classifier
intent_classifier = LogisticRegression()
intent_classifier.fit(X_train, training_labels)

# Emotion detection function
def detect_emotion(text):
    results = emotion_classifier(text)
    if results and len(results) > 0:
        return results[0]['label'].lower()
    else:
        return "neutral"

# Intent detection function
def detect_intent(text):
    processed_text = preprocess_text(text)
    X_test = vectorizer.transform([processed_text])
    prediction = intent_classifier.predict(X_test)
    return prediction[0]

@app.route("/")
def home():
    return send_from_directory('', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('', path)

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion_endpoint():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    emotion = detect_emotion(text)
    intent = detect_intent(text)
    return jsonify({"emotion": emotion, "intent": intent})

# Session logs
session_logs_file = "session_logs.json"

def save_session_log(entry):
    try:
        logs = []
        if os.path.exists(session_logs_file):
            with open(session_logs_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(session_logs_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Error saving session log: {e}")

def generate_local_response(prompt, max_new_tokens=100):
    encoded = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove the prompt from the output to get only the generated continuation
    return output_text[len(prompt):].strip()

@app.route("/converse", methods=["POST"])
def converse():
    data = request.json
    user_input = data.get("text", "")
    conversation_history = data.get("history", [])

    if not user_input:
        print("ERROR: No user input provided.")
        return jsonify({"error": "No text provided"}), 400

    user_emotion = detect_emotion(user_input)
    user_intent = detect_intent(user_input)

    system_prompt = (
        "You are EmotiCare, an empathetic and supportive AI assistant specialized in emotional wellness. "
        "Your goal is to provide compassionate, understanding, and helpful responses tailored to the user's emotional state and intent. "
        "Avoid generic or dismissive replies. Use a warm and encouraging tone. "
        f"The user's current emotional state is: {user_emotion}. "
        f"The user's intent is: {user_intent}. "
        "Incorporate this context to respond appropriately and offer practical advice or comforting words."
    )

    limited_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

    messages = [{"role": "system", "content": system_prompt}]
    for entry in limited_history:
        if "role" in entry and "content" in entry:
            messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": user_input})

    print("\n[DEBUG] Sending messages to OpenAI or local model:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content']}")

    if openai_api_key:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=400,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
            print(f"[DEBUG] Response received from OpenAI: {reply}")

            if not reply:
                reply = "I'm here to listen and support you. Could you please tell me more about how you're feeling?"

            return jsonify({"response": reply})
        except Exception as e:
            error_message = str(e)
            if "insufficient_quota" in error_message or "429" in error_message:
                user_friendly_message = (
                    "OpenAI API quota exceeded. Please check your OpenAI plan and billing details. "
                    "For more information, visit https://platform.openai.com/docs/guides/error-codes/api-errors."
                )
                print(f"[ERROR] OpenAI API quota exceeded: {error_message}")
                return jsonify({"error": user_friendly_message}), 429
            elif "invalid_api_key" in error_message or "401" in error_message:
                user_friendly_message = (
                    "Invalid OpenAI API key provided. Please check your API key and update it accordingly. "
                    "You can find your API key at https://platform.openai.com/account/api-keys."
                )
                print(f"[ERROR] OpenAI API invalid key: {error_message}")
                # Fallback to local model
                prompt_text = (
                    "You are EmotiCare, an empathetic and supportive AI assistant specialized in emotional wellness.\n"
                    "Your goal is to provide compassionate, understanding, and helpful responses tailored to the user's emotional state and intent.\n"
                    "Avoid generic or dismissive replies. Use a warm and encouraging tone.\n"
                    f"The user's current emotional state is: {user_emotion}.\n"
                    f"The user's intent is: {user_intent}.\n"
                    "Incorporate this context to respond appropriately and offer practical advice or comforting words.\n\n"
                    f"User: {user_input}\n"
                    "Assistant:\nPlease respond empathetically, supportively, and contextually in a concise manner."
                )
                reply = generate_local_response(prompt_text, max_length=100)
                if not reply:
                    reply = "I'm here to listen and support you. Could you please tell me more about how you're feeling?"
                print(f"[DEBUG] Response generated by local model (fallback due to invalid API key): {reply}")
                return jsonify({"response": reply, "warning": user_friendly_message}), 200
            else:
                print(f"[ERROR] OpenAI API error: {error_message}")
                return jsonify({"error": error_message}), 500
    else:
        # Use local model fallback
        prompt_text = (
            "You are EmotiCare, an empathetic and supportive AI assistant specialized in emotional wellness.\n"
            "Your goal is to provide compassionate, understanding, and helpful responses tailored to the user's emotional state and intent.\n"
            "Avoid generic or dismissive replies. Use a warm and encouraging tone.\n"
            f"The user's current emotional state is: {user_emotion}.\n"
            f"The user's intent is: {user_intent}.\n"
            "Incorporate this context to respond appropriately and offer practical advice or comforting words.\n\n"
            f"User: {user_input}\n"
            "Assistant:\nPlease respond empathetically, supportively, and contextually in a concise manner."
        )
        reply = generate_local_response(prompt_text, max_length=100)
        if not reply:
            reply = "I'm here to listen and support you. Could you please tell me more about how you're feeling?"
        print(f"[DEBUG] Response generated by local model: {reply}")
        return jsonify({"response": reply})
    reply = generate_local_response(prompt_text, max_new_tokens=100)

@app.route("/generate_wellness_content", methods=["POST"])
def generate_wellness_content():
    data = request.json
    print(f"[DEBUG] Request received: {data}")

    content_type = data.get("type", "meditation")
    user_context = data.get("context", "")

    prompt_map = {
        "meditation": "Generate a detailed and calming short guided meditation to help someone relax, focus on breathing, and feel peaceful.",
        "story": "Generate an uplifting and motivational story that inspires hope and resilience.",
        "quote": "Generate an original inspirational quote about strength, resilience, and hope.",
        "coping_strategy": "Generate a personalized and practical coping strategy for managing anxiety and stress, including actionable steps."
    }

    prompt = prompt_map.get(content_type, prompt_map["meditation"])
    if user_context:
        prompt += f" Context: {user_context}"

    print(f"[DEBUG] Final prompt: {prompt}")

    if openai_api_key:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in wellness content generation."},
                {"role": "user", "content": prompt}
            ]

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300,
                temperature=0.85,
            )
            content = response.choices[0].message.content.strip()
            print(f"[DEBUG] Generated content: {content}")

            return jsonify({"content": content})
        except Exception as e:
            error_message = str(e)
            if "insufficient_quota" in error_message or "429" in error_message:
                user_friendly_message = (
                    "OpenAI API quota exceeded. Please check your OpenAI plan and billing details. "
                    "For more information, visit https://platform.openai.com/docs/guides/error-codes/api-errors."
                )
                print(f"[ERROR] OpenAI content generation quota exceeded: {error_message}")
                return jsonify({"error": user_friendly_message}), 429
            elif "invalid_api_key" in error_message or "401" in error_message:
                user_friendly_message = (
                    "Invalid OpenAI API key provided. Please check your API key and update it accordingly. "
                    "You can find your API key at https://platform.openai.com/account/api-keys."
                )
                print(f"[ERROR] OpenAI content generation invalid key: {error_message}")
                # Fallback to local model
                prompt_text = "You are a helpful assistant specialized in wellness content generation.\nUser: " + prompt + "\nAssistant:"
                content = generate_local_response(prompt_text)
                if not content:
                    content = "Here is some wellness advice: Take a few deep breaths and focus on the present moment."
                print(f"[DEBUG] Content generated by local model (fallback due to invalid API key): {content}")
                return jsonify({"content": content, "warning": user_friendly_message}), 200
            else:
                print(f"[ERROR] OpenAI content generation failed: {error_message}")
                return jsonify({"error": error_message}), 500
    else:
        # Use local model fallback
        prompt_text = "You are a helpful assistant specialized in wellness content generation.\nUser: " + prompt + "\nAssistant:"
        content = generate_local_response(prompt_text)
        if not content:
            content = "Here is some wellness advice: Take a few deep breaths and focus on the present moment."
        print(f"[DEBUG] Content generated by local model: {content}")
        return jsonify({"content": content})

# Facial emotion detection endpoint
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def simple_facial_emotion_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return "no_face_detected"
    else:
        return "neutral"

@app.route("/facial_emotion_detection", methods=["POST"])
def facial_emotion_detection():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
    emotion_label = simple_facial_emotion_detection(img)
    return jsonify({"emotion": emotion_label})

if __name__ == "__main__":
    app.run(debug=True)
