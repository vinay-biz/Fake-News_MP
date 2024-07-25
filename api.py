from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open('text_clf_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    user_input = data['text']
    
    # The model pipeline handles vectorization internally
    prediction = model.predict([user_input])
    result = "Fake News" if prediction[0] == 1 else "Real News"
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
