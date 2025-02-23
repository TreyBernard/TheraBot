from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (if your React runs on a different port)


API_KEY = "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"

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

    prompt = f"""
    You are an empathetic AI therapist. The user has shared:
    - Name: {user_info.get('name', 'Unknown')}
    - Age: {user_info.get('age', 'Unknown')}
    - School: {user_info.get('school', 'Unknown')}
    - Current Feeling: {user_info.get('feeling', 'Unknown')}
    - Detected Emotion: {detected_emotion}
    
    The user says: "{user_message}"
    
    Provide a thoughtful and compassionate therapeutic response.
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
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
        
        print("DEBUG API RESPONSE:", response_json)  # Print full API response in Flask logs
        
        return jsonify({"response": response_json["choices"][0]["message"]["content"]})
    except Exception as e:
        print("ERROR:", str(e))  # Print error in Flask logs
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    # By default, runs on http://127.0.0.1:5000
    app.run(debug=True)
