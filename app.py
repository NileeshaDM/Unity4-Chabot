from flask import Flask, render_template, request, jsonify
import chat  # Import your chatbot script

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chat.html', messages=[])

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()  # Retrieve JSON data
    user_message = data.get('user_message', '')  # Get the 'user_message' key

    print(f"Received user message: {user_message}")

    bot_response = chat.chatbot_response(user_message)  # Use your chatbot script here

    print(f"Chatbot response: {bot_response}")
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
