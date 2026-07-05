from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    life = float(request.form['life'])
    mean = float(request.form['mean'])
    expected = float(request.form['expected'])
    gni = float(request.form['gni'])

    features = np.array(
        [[life, mean, expected, gni]]
    )

    prediction = model.predict(features)

    result = prediction[0]

    return render_template(
        'index.html',
        prediction_text=f"HDI Category: {result}"
    )


if __name__=="__main__":
    app.run(debug=True)