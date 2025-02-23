from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
from FBED import predict

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (if your React runs on a different port)

chat_sessions = {}
API_KEY = "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"

is_talking = False
label_predicted = None

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
    
# Store talking state per user
talking_states = {}

# Endpoint for SER.py to get is_talking
@app.route("/get_talking_status", methods=["GET"])
def get_talking_status():
    print("Checking talking status")
    print(f"is_talking: {is_talking}")
    return jsonify({"isTalking": is_talking})

# Endpoint for SER.py to send prediction back
@app.route("/receive_prediction", methods=["POST"])
def receive_prediction():
    global label_predicted
    data = request.get_json()
    label_predicted = data.get("label_predicted", "Unknown")
    print(f"Received Prediction: {label_predicted}")
    return jsonify({"message": "Prediction received successfully"})

# Endpoint to update is_talking (optional)
@app.route("/update_talking_status", methods=["POST"])
def update_talking_status():
    global is_talking
    data = request.get_json()
    is_talking = data.get("isTalking", False)
    return jsonify({"message": "Updated is_talking status"})

def emotion_predict():
    emotion_prediction = predict()
    return emotion_prediction

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    user_info = data.get("user_info", {})
    detected_emotion = data.get("emotion", emotion_predict() or "neutral")
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
    By taking into account the user's emotions through facial and body expressions, attempt in using Cognitive 
    Behavioral Therapy (CBT) to better understand the patients feelings and goals coming out of this session. 
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
                submitted by the user, like in the form of resources, things to do in the school area, etc. By taking into account the user's emotions through facial
                and body expressions, attempt in using Cognitive Behavioral Therapy (CBT) to better understand the patients feelings and goals coming out of this session. 
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

