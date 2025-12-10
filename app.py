import os
import threading
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# -------------------------------
# Load environment variables
load_dotenv()
app = Flask(__name__)

# -------------------------------
# Azure Speech Config
speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "en-US"
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

# Thread-safe flag for speaking state
speech_lock = threading.Lock()
is_speaking = False

# Register the speech completion event handler DIRECTLY on the synthesizer
def _on_speech_synth_completed(evt):
    global is_speaking
    with speech_lock:
        is_speaking = False
    print("✅ Speech finished (via event)")

synthesizer.synthesis_completed.connect(_on_speech_synth_completed)
synthesizer.synthesis_canceled.connect(_on_speech_synth_completed)

# -------------------------------
# Azure OpenAI Config
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Conversation memory
conversation_history = [
    {"role": "system", "content": "You are a helpful talking avatar assistant."}
]

# -------------------------------
# Speech to Text
def speech_to_text_microphone():
    try:
        audio_config = speechsdk.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized.")
            return ""
        elif result.reason == speechsdk.ResultReason.Canceled:
            print("Speech Recognition canceled: {}".format(result.cancellation_details.reason))
            return ""
        return ""
    except Exception as err:
        print("Speech to text error:", err)
        return ""

# -------------------------------
# Text to Speech (no callback on returned future)
def text_to_speech(text):
    global is_speaking
    with speech_lock:
        if is_speaking:
            print("🛑 Stopping previous speech (if any).")
            stop_speaking()
        is_speaking = True
        try:
            synthesizer.speak_text_async(text)
        except Exception as e:
            print("❌ Synthesizer error:", e)
            is_speaking = False

def stop_speaking():
    global is_speaking
    with speech_lock:
        try:
            if is_speaking:
                print("🔸 Calling stop_speaking_async()")
                synthesizer.stop_speaking_async()
                is_speaking = False
                print("✅ stop_speaking_async() called (stop may not be instant due to SDK/OS buffer)")
        except Exception as e:
            print("❌ Stop error:", e)

# -------------------------------
# AI Brain
def get_ai_reply(user_text):
    global conversation_history
    command = user_text.lower().strip()
    if command == "stop":
        stop_speaking()
        return "Okay, I stopped."
    if command in ["new conversation", "reset", "start over"]:
        conversation_history.clear()
        conversation_history.append({"role": "system", "content": "You are a helpful talking avatar assistant."})
        return "Conversation reset."
    if command == "exit":
        stop_speaking()
        conversation_history.clear()
        conversation_history.append({"role": "system", "content": "You are a helpful talking avatar assistant."})
        return "Goodbye."
    conversation_history.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=conversation_history
    )
    ai_reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": ai_reply})
    return ai_reply

# -------------------------------
# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/listen", methods=["POST"])
def listen():
    user_text = speech_to_text_microphone()
    reply = get_ai_reply(user_text)
    text_to_speech(reply)
    return jsonify({"user_text": user_text, "reply": reply})

@app.route("/type", methods=["POST"])
def typed_text():
    data = request.get_json(force=True)
    user_text = data.get("text", "")
    reply = get_ai_reply(user_text)
    text_to_speech(reply)
    return jsonify({"user_text": user_text, "reply": reply})

@app.route("/stop", methods=["POST"])
def stop_audio():
    stop_speaking()
    return jsonify({"status": "stopped"})

# -------------------------------
if __name__ == "__main__":
    print("✅ AI Avatar App Running...")
    app.run(debug=True)
