import os
import glob
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ------- Load env and Flask setup ---------
load_dotenv()
app = Flask(__name__)
last_reply = ""

# Azure Speech setup
speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "en-US"
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

# Azure OpenAI/LangChain setup
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")

# ------ Build VectorDatabase at startup ------
def build_vector_db():
    docs = []
    for fname in glob.glob("knowledge_base/*.txt"):
        with open(fname, encoding="utf-8") as f:
            docs.append(f.read())
    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in docs:
        chunks.extend([c for c in splitter.split_text(doc) if c.strip()])
    embeddings = AzureOpenAIEmbeddings(
            model=EMBEDDING_DEPLOYMENT_NAME,
            azure_deployment=EMBEDDING_DEPLOYMENT_NAME,  # <- Must be embedding deployment ONLY (not GPT chat model)
            azure_endpoint=OPENAI_ENDPOINT,
            api_version=OPENAI_VERSION,
            api_key=OPENAI_KEY,
        )
    vector_db = FAISS.from_texts(chunks, embeddings)
    print(f"[RAG] Vector DB loaded with {len(chunks)} chunks.")
    return vector_db

vector_db = build_vector_db()
llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version=OPENAI_VERSION,
    api_key=OPENAI_KEY,
    temperature=1,
)

# --- Speech to Text ---
def speech_to_text_microphone():
    try:
        audio_config = speechsdk.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        print("[STT] Listening for speech...")
        result = recognizer.recognize_once()
        print("[STT] Recognized:", result.text)
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        else:
            return ""
    except Exception as err:
        print("[STT] Error:", err)
        return ""

# --- Text to Speech ---
# def text_to_speech(text):
#     print("[TTS] Speaking:", text)
#     try:
#         synthesizer.speak_text_async(text)
#     except Exception as e:
#         print("[TTS] Error:", e)

def text_to_speech(text, full=False):
    """
    Read text aloud. If full=False, only read first 2 sentences.
    """
    if not full:
        # Split text into sentences and take first 2
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 2:
            # Read first 2 sentences and add prompt
            preview_text = '. '.join(sentences[:2]) + '. Would you like to hear more?'
            print(f"[TTS] Speaking preview: {preview_text}")
        else:
            preview_text = text
            print(f"[TTS] Speaking: {preview_text}")
    else:
        preview_text = text
        print(f"[TTS] Speaking full text: {preview_text}")
    
    try:
        synthesizer.speak_text_async(preview_text)
    except Exception as e:
        print("[TTS] Error:", e)

# --- RAG QA using LangChain VectorDB and LLM ---
def get_ai_reply(user_text):
    command = user_text.lower().strip()
    if command in {"stop", "exit"}:
        return "Okay, stopping." if command == "stop" else "Goodbye."
    if command in {"new conversation", "reset", "start over"}:
        return "Conversation reset."
    if vector_db is None:
        return "Sorry, no knowledge base is loaded."
    try:
        retrieved_docs = vector_db.similarity_search(user_text, k=4)
    except Exception as e:
        print(f"[RAG] Error during similarity search: {e}")
        return "Trouble accessing the search database."
    kb_context = "\n".join([doc.page_content for doc in retrieved_docs])
    print(f"[RAG] Context:\n{kb_context[:500]}")
    prompt = (
        f"Use the following knowledge base context to answer the user's question. "
        f"If the answer is not in the context, admit it. Cite facts from the context.\n"
        f"Context:\n{kb_context}\n\n"
        f"User: {user_text}\nAssistant:"
    )
    try:
        ai_reply = llm.invoke(prompt).content  # Changed from predict() to invoke()
        print("[LLM] AI replied:", ai_reply)
        return ai_reply
    except Exception as e:
        print("OpenAI API error:", e)
        return "Sorry, I couldn't generate a reply."

# ------- Flask routes ---------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/listen", methods=["POST"])
def listen():
    global last_reply
    user_text = speech_to_text_microphone()
    if not user_text:
        return jsonify({"user_text": "", "reply": "Sorry, I didn't hear anything."})
    
    # Clean the text - Azure Speech adds punctuation like "Yes."
    cleaned_text = user_text.lower().strip().rstrip('.!?')
    print(f"[DEBUG] Original: '{user_text}' | Cleaned: '{cleaned_text}'")
    
    # Check if user wants to hear more
    if cleaned_text in ["yes", "yes please", "hear more", "read more", "continue", "more", "yeah", "yep", "sure"]:
        if last_reply:
            print(f"[DEBUG] Playing full reply")
            text_to_speech(last_reply, full=True)
            return jsonify({"user_text": user_text, "reply": last_reply, "full": True})
        else:
            print(f"[DEBUG] No previous reply to play")
    
    reply = get_ai_reply(user_text)
    last_reply = reply
    text_to_speech(reply, full=False)
    return jsonify({"user_text": user_text, "reply": reply, "preview": True})

@app.route("/type", methods=["POST"])
def typed_text():
    global last_reply
    data = request.get_json(force=True)
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"user_text": "", "reply": "Empty message."})
    
    # Check if user wants to hear more
    if user_text.lower().strip() in ["yes", "yes please", "hear more", "read more", "continue", "more"]:
        if last_reply:
            text_to_speech(last_reply, full=True)
            return jsonify({"user_text": user_text, "reply": last_reply, "full": True})
    
    reply = get_ai_reply(user_text)
    last_reply = reply
    text_to_speech(reply, full=False)
    return jsonify({"user_text": user_text, "reply": reply, "preview": True})

@app.route("/read_more", methods=["POST"])
def read_more():
    """Route to read the full reply"""
    global last_reply
    if last_reply:
        text_to_speech(last_reply, full=True)
        return jsonify({"status": "reading_full", "reply": last_reply})
    else:
        return jsonify({"status": "no_reply", "reply": "No previous reply to read."})

@app.route("/stop", methods=["POST"])
def stop_audio():
    print("Received /stop; no operation performed.")
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    print("✅ AI Avatar RAG App Running...")
    app.run(debug=True)