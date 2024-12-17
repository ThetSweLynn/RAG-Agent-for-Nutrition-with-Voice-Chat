from flask import Flask, request, jsonify
from nutrition_rag import call_rag_agent, listen_for_audio, speak_text

app = Flask(__name__)

# Define route for text query
@app.route('/query', methods=['POST'])
def query_rag():
    data = request.get_json()  # Get the JSON data from the request
    user_query = data.get('query', '')  # Extract the query
    
    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    
    # Call RAG agent
    response = call_rag_agent(user_query)
    
    return jsonify({"response": response}),200

# Define route for speech input (audio query)
@app.route('/audio_query', methods=['POST'])
def audio_query():
    audio_file = request.files.get('audio')  # Get the audio file from the request
    
    if not audio_file:
        return jsonify({"error": "Audio file is required"}), 400
    
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)
    
    query = listen_for_audio()  # Process the audio file and get the query text
    
    if query:
        # Call RAG agent for the query
        response = call_rag_agent(query)
        speak_text(response)  # Convert the response to speech
        return jsonify({"response": response})
    else:
        return jsonify({"error": "Could not recognize speech."}), 400

if __name__ == '__main__':
    app.run(debug=True)
