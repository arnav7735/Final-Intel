import tkinter as tk
import subprocess
import os
import whisper
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import random

random_number = random.randint(1000, 9999)
# Paths to your model files
TRANSCRIPTION_FILE = f"C:\\Users\\Arnav\\Desktop\\kpr_transcription_{random_number}.txt"
CHUNKS_FILE = f"C:\\Users\\Arnav\\Desktop\\kpr_chunks_{random_number}.pkl"
INDEX_FILE = f"C:\\Users\\Arnav\\Desktop\\kpr_index_{random_number}.faiss"
EMBEDDINGS_FILE = f"C:\\Users\\Arnav\\Desktop\\kpr_embeddings_{random_number}.npy"

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Global variables
ffmpeg_process = None
output_text = ""  # To store the output from the model

def create_chunks(transcription_text):
    # Split the transcription text into chunks (you can customize the chunk size)
    chunk_size = 500  # For example, 500 characters per chunk
    chunks = [transcription_text[i:i + chunk_size] for i in range(0, len(transcription_text), chunk_size)]
    
    # Save chunks to file
    with open(CHUNKS_FILE, 'wb') as f:
        pickle.dump(chunks, f)

    return chunks

def create_faiss_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)  # Add embeddings to the index

    # Save FAISS index and embeddings
    faiss.write_index(index, INDEX_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)

    print("FAISS index and embeddings created successfully.")

def start_system_audio_recording():
    global ffmpeg_process
    output_filename = 'system_audio_output.wav'  # Output file for system audio
    command = [
        'ffmpeg',
        '-y',
        '-f', 'dshow',
        '-i', 'audio=Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)',  # Modify this line to match your system audio device
        output_filename
    ]
    
    print("Starting system audio recording...")
    ffmpeg_process = subprocess.Popen(command)

def stop_system_audio_recording():
    global ffmpeg_process
    if ffmpeg_process:
        print("Stopping system audio recording...")
        ffmpeg_process.terminate()
        ffmpeg_process.wait()  # Wait for the process to finish
        ffmpeg_process = None
        print("System audio recording stopped and saved.")

        # Get the user prompt from the input field
        user_query = prompt_input.get("1.0", tk.END).strip()  # Retrieve the user prompt
        if not user_query:
            user_query = "Give me a detailed summary of what is happening here"  # Default prompt if none provided

        # Call the function to process audio after recording stops
        process_audio_with_model('system_audio_output.wav', user_query)


def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcription_text = result['text']

    # Save the transcription to a file
    with open(TRANSCRIPTION_FILE, 'w', encoding='utf-8') as f:
        f.write(transcription_text)

    return transcription_text

def load_transcription(transcription_file):
    if not os.path.exists(transcription_file):
        print(f"Transcription file '{transcription_file}' does not exist.")
        return None
    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            transcription = f.read()
        print("Transcription loaded successfully.")
        return transcription
    except Exception as e:
        print(f"Error loading transcription: {e}")
        return None

def load_chunks(chunks_file):
    if not os.path.exists(chunks_file):
        print(f"Chunks file '{chunks_file}' does not exist.")
        return None
    try:
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
        print("Chunks loaded successfully.")
        return chunks
    except Exception as e:
        print(f"Error loading chunks: {e}")
        return None

def load_faiss_index(index_file, embeddings_file):
    if not os.path.exists(index_file) or not os.path.exists(embeddings_file):
        print(f"FAISS index file '{index_file}' or embeddings file '{embeddings_file}' does not exist.")
        return None, None
    try:
        index = faiss.read_index(index_file)
        embeddings = np.load(embeddings_file)
        print("FAISS index and embeddings loaded successfully.")
        return index, embeddings
    except Exception as e:
        print(f"Error loading FAISS index or embeddings: {e}")
        return None, None

def load_chat_model(google_api_key):
    if not google_api_key:
        print("Google API Key is not set.")
        return None
    try:
        chat = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.4,
            google_api_key=google_api_key,
            convert_system_message_to_human=True
        )
        print("Chat model loaded successfully.")
        return chat
    except Exception as e:
        print(f"Error loading ChatGoogleGenerativeAI model: {e}")
        return None

def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    if index is None or chunks is None:
        print("FAISS index or chunks are not loaded.")
        return []
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        _, I = index.search(query_embedding, top_k)
        retrieved = [chunks[i] for i in I[0]]
        print(f"Retrieved {len(retrieved)} relevant chunks.")
        return retrieved
    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        return []

def answer_question(chat, retrieved_chunks, user_query):
    if chat is None or not retrieved_chunks:
        print("Chat model is not loaded or no chunks retrieved.")
        return None
    try:
        context = " ".join(retrieved_chunks)
        message = [
            SystemMessage(content=f"Use the following context to answer the question in detail:\n\n{context}"),
            HumanMessage(content=user_query)
        ]
        result = chat.invoke(message)
        print("Answer generated successfully.")
        return result
    except Exception as e:
        print(f"Error generating answer: {e}")
        return None

def process_audio_with_model(system_audio, user_query):
    # Check if the transcription file already exists
    if os.path.exists(TRANSCRIPTION_FILE):
        print(f"Transcription file already exists: {TRANSCRIPTION_FILE}")
        # Load transcription
        with open(TRANSCRIPTION_FILE, 'r', encoding='utf-8') as f:
            transcription = f.read()
    else:
        # If it doesn't exist, transcribe audio
        transcription = transcribe_audio(system_audio)
        if not transcription:
            print("Transcription failed.")
            return  # Exit if transcription fails

        # Create chunks and FAISS index
        chunks = create_chunks(transcription)
        create_faiss_index(chunks)

    print(f"Transcription: {transcription}")

    # Load other necessary model components
    chunks = load_chunks(CHUNKS_FILE)
    index, embeddings = load_faiss_index(INDEX_FILE, EMBEDDINGS_FILE)
    chat = load_chat_model(GOOGLE_API_KEY)

    if chunks and index and chat:
        retrieved_chunks = retrieve_relevant_chunks(user_query, index, chunks)
        response = answer_question(chat, retrieved_chunks, user_query)

        if response:
            output_text = response.content
            print(f"\n--- Answer ---\n{output_text}\n")
            update_output_display(output_text)
        else:
            print("Failed to generate an answer.")
    else:
        print("Model components not loaded properly.")

# Function to update the GUI with the model output
def update_output_display(output):
    output_display.delete(1.0, tk.END)  # Clear previous output
    output_display.insert(tk.END, output)  # Insert new output

# Initialize GUI
root = tk.Tk()
root.title("Audio Recorder and Summarizer")
root.geometry("400x400")
root.configure(bg="#282c34")

# Create frames for the pages
home_frame = tk.Frame(root, bg="#282c34")
audio_frame = tk.Frame(root, bg="#282c34")

# Home Page Elements
home_label = tk.Label(home_frame, text="Welcome to the Summarizer", font=("Helvetica", 18), fg="white", bg="#282c34")
home_label.pack(pady=20)

intro_label = tk.Label(home_frame, text="This application records system audio and summarizes it", font=("Helvetica", 12), fg="white", bg="#282c34")
intro_label.pack(pady=10)

# Audio Recording Page Elements
title_label = tk.Label(audio_frame, text="Audio Recorder", font=("Helvetica", 18), fg="white", bg="#282c34")
title_label.pack(pady=10)

start_button = tk.Button(audio_frame, text="Start System Audio", command=start_system_audio_recording, font=("Helvetica", 14), fg="white", bg="#4CAF50")
start_button.pack(pady=5)

stop_button = tk.Button(audio_frame, text="Stop System Audio", command=stop_system_audio_recording, font=("Helvetica", 14), fg="white", bg="#f44336")
stop_button.pack(pady=5)

prompt_label = tk.Label(audio_frame, text="Enter your prompt:", font=("Helvetica", 12), fg="white", bg="#282c34")
prompt_label.pack(pady=5)

prompt_input = tk.Text(audio_frame, height=4, width=40, font=("Helvetica", 12))
prompt_input.pack(pady=5)

# Function to handle the submission of user input
def submit_prompt():
    user_query = prompt_input.get("1.0", tk.END).strip()  # Get the user input prompt
    if not user_query:
        user_query = "Give me a detailed summary of what is happening here"  # Default prompt if none provided
    
    # Call the function to process the audio with the model
    process_audio_with_model('system_audio_output.wav', user_query)

# Add a Submit Button
submit_button = tk.Button(audio_frame, text="Submit", command=submit_prompt, font=("Helvetica", 14), fg="white", bg="#4CAF50")
submit_button.pack(pady=5)

output_display = tk.Text(audio_frame, height=10, width=50, font=("Helvetica", 12))
output_display.pack(pady=5)

# Set the home frame as the initial view
home_frame.pack(expand=True, fill="both")

def switch_to_audio_frame():
    home_frame.pack_forget()
    audio_frame.pack(expand=True, fill="both")

# Add a button to switch to the audio frame
switch_button = tk.Button(home_frame, text="Go to Audio Recorder", command=switch_to_audio_frame, font=("Helvetica", 14), fg="white", bg="#2196F3")
switch_button.pack(pady=20)

# Run the application
root.mainloop()
