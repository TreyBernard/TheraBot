from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (if your React runs on a different port)

API_KEY = "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"
chat_sessions = {}
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Simple mock check
    if username == 'admin' and password == 'secret':
        return jsonify({'success': True, 'message': 'Login successful!'})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials.'}), 401

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    user_info = data.get("user_info", {})
    detected_emotion = data.get("emotion", "neutral")
    user_message = data.get("message", "")
    username = user_info.get("name", "Unknown")
    
    prompt = f"""
    You are an empathetic AI therapist. The user has shared:
    - Name: {user_info.get('name', 'Unknown')}
    - Age: {user_info.get('age', 'Unknown')}
    - School: {user_info.get('school', 'Unknown')}
    - Current Feeling: {user_info.get('feeling', 'Unknown')}
    - Detected Emotion: {detected_emotion}
    
    The user says: "{user_message}"
    
    Provide a thoughtful and compassionate therapeutic response. Provide useful feedback when appropriate
    based on the school submitted by the user, like in the form of resources, things to do in the school area, etc.
    """

    if username not in chat_sessions:
        chat_sessions[username] = [
            {"role": "system", "content": f"""
                You are an empathetic AI therapist.
                The user has shared:
                - Name: {user_info.get('name', 'Unknown')}
                - Age: {user_info.get('age', 'Unknown')}
                - School: {user_info.get('school', 'Unknown')}
                - Current Feeling: {user_info.get('feeling', 'Unknown')}
                - Detected Emotion: {detected_emotion}
                
                Engage in a thoughtful conversation, remembering past exchanges within this session. Also, provide useful feedback when appropriate based on the school 
                submitted by the user, like in the form of resources, things to do in the school area, etc.
            """}
        ]

    chat_sessions[username].append({"role": "user", "content": user_message})

    # Keep only the last 5 exchanges (rolling memory)
    chat_sessions[username] = chat_sessions[username][-7:]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    print("DEBUG: API Key being used ->", API_KEY)  # Check if API key is set


    data = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": chat_sessions[username],
        "temperature": 1.0,
        "top_p": 1.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "repetition_penalty": 1,
        "top_k": 0
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        response_json = response.json()
        ai_response = response_json["choices"][0]["message"]["content"]
        chat_sessions[username].append({"role": "assistant", "content": ai_response})
        
        print("DEBUG API RESPONSE:", response_json)  # Print full API response in Flask logs
        
        return jsonify({"response": response_json["choices"][0]["message"]["content"]})
    except Exception as e:
        print("ERROR:", str(e))  # Print error in Flask logs
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    # By default, runs on http://127.0.0.1:5000
    app.run(debug=True)