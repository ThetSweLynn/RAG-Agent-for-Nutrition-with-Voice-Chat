import os
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pygame import mixer
from datetime import date
import nutrition_rag  # Assuming this is the file that contains your RAG logic

mixer.init()

today = str(date.today())

# text to speech function
def speak_text(text):
    try:
        # Preprocess the text to handle special characters and length
        cleaned_text = text.replace('\n', ' ').strip()  # Remove newlines and extra spaces
        if len(cleaned_text) > 200:  # Adjust the maximum length as needed
            chunks = [cleaned_text[i:i+200] for i in range(0, len(cleaned_text), 200)]
            for chunk in chunks:
                tts = gTTS(text=chunk, lang='en-US')  # Adjust language if needed
                with BytesIO() as mp3_file:
                    tts.write_to_fp(mp3_file)
                    mp3_file.seek(0)
                    mixer.music.load(mp3_file, "mp3")
                    mixer.music.play()
                    while mixer.music.get_busy():
                        pass
        else:
            tts = gTTS(text=cleaned_text, lang='en-US')  # Adjust language if needed
            with BytesIO() as mp3_file:
                tts.write_to_fp(mp3_file)
                mp3_file.seek(0)
                mixer.music.load(mp3_file, "mp3")
                mixer.music.play()
                while mixer.music.get_busy():
                    pass
    except Exception as e:
        print(f"Error speaking text: {e}")

# Save conversation to a log file 
def append2log(text):
    global today
    fname = 'chatlog-' + today + '.txt'
    with open(fname, "a", encoding='utf-8') as f:
        f.write(text + "\n")

# Function to handle speech recognition
def listen_for_audio():
    rec = sr.Recognizer()
    mic = sr.Microphone()
    rec.dynamic_energy_threshold = False
    rec.energy_threshold = 400
    
    with mic as source:
        rec.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening for audio input...")
        audio = rec.listen(source, timeout=30, phrase_time_limit=30)
        
    try:
        request = rec.recognize_google(audio, language="en-EN")
        return request
    except Exception as e:
        print("Error recognizing speech:", e)
        return None

# Main function
def main():
    global today, model, chat, slang 

    # Ask the user to choose between audio or text input
    choice = input("Choose input method: (1) Audio (2) Text\n")
    
    if choice == "1":
        # Audio input
        while True:
            print("Please speak your query:")
            query = listen_for_audio()
            if query:
                print(f"You: {query}")
                append2log(f"You: {query}")
                
                # Pass the request to the RAG agent
                response = nutrition_rag.call_rag_agent(query)  # Assuming call_rag_agent is defined in nutrition_rag
                print(f"AI: {response}")
                speak_text(response.replace("*", ""))  # Speaking the response
                append2log(f"AI: {response}")
            else:
                print("Sorry, I couldn't understand your speech.")
                
    elif choice == "2":
        # Text input
        while True:
            query = input("Please type your query: ")
            if query.lower() == "exit":
                print("Exiting...")
                break
            append2log(f"You: {query}")
            
            # Pass the request to the RAG agent
            response = nutrition_rag.call_rag_agent(query)  # Assuming call_rag_agent is defined in nutrition_rag
            print(f"AI: {response}")
            speak_text(response.replace("*", ""))  # Speaking the response
            append2log(f"AI: {response}")
            
    else:
        print("Invalid choice. Please restart the program and choose a valid option.")

if __name__ == "__main__":
    main()