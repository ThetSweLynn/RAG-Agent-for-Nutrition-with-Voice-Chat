#chcp 65001
#Type this ⬆️ first before running if you are using Windows 
#⬆️⬆️ This is just for Chinese characters to display properly in the console


# Google Gemini-Powered Real-Time Language Translator with Audio
# Tested and working on Windows 11. 
# By TechMakerAI on YouTube
# 
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pygame import mixer
from datetime import date
import nutrition_rag
 
mixer.init()
#os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

today = str(date.today())
 
# text to speech function
''' 
def speak_text(text):
    
    mp3audio = BytesIO() 
    
    tts = gTTS(text, lang='en-US', tld = 'us')     
    
    tts.write_to_fp(mp3audio)

    mp3audio.seek(0)

    mixer.music.load(mp3audio, "mp3")
    mixer.music.play()

    while mixer.music.get_busy():
        pass
    
    mp3audio.close()
'''

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
        

# save conversation to a log file 
def append2log(text):
    global today
    fname = 'chatlog-' + today + '.txt'
    with open(fname, "a", encoding='utf-8') as f:
        f.write(text + "\n")
        f.close 

# define default language to work with the AI model 
slang = "en-EN"

# Main function  
def main():
    global today, model, chat, slang 
    
    rec = sr.Recognizer()
    mic = sr.Microphone()
    rec.dynamic_energy_threshold=False
    rec.energy_threshold = 400    
    
    # while loop for interaction with AI model  
 
    while True:     
        
        with mic as source1:            
            rec.adjust_for_ambient_noise(source1, duration= 0.5)

            print("Listening ...")
            
            audio = rec.listen(source1, timeout = 30, phrase_time_limit = 30)
            
            try:                 
               
                request = rec.recognize_google(audio, language=slang )
                #request = rec.recognize_wit(audio, key=wit_api_key )
                
                if len(request) < 2:
                    continue 
                    
                if "that's all" in request.lower():
                                               
                    append2log(f"You: {request}\n")
                        
                    speak_text("Bye now")
                        
                    append2log(f"AI: Bye now. \n")                        

                    print('Bye now')

                    continue
                                       
                # Send user input to Gemini model and receive response
                append2log(f"You: {request}\n")
                print(f"You: {request}\nAI: ")
                
                #Pass the request to our RAG agent
                response = nutrition_rag.call_rag_agent(request)  # This is where the magic happens
                print(response)
                speak_text(response.replace("*", ""))
                append2log(f"AI: {response}\n")
                
                
            except Exception as e:
                #print(response)
                continue 
 
if __name__ == "__main__":
    main()



