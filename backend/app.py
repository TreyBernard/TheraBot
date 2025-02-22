from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == "__main__":
    # Run the Flask server (by default on http://127.0.0.1:5000)
    app.run(debug=True)