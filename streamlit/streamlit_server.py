import pickle

from flask import Flask, jsonify, request

model_file = './models/pipeline.bin'

with open(model_file, 'rb') as f_in:
    Pipeline = pickle.load(f_in)

app = Flask('serve')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    y_pred = Pipeline.predict_proba(customer)[0, 1]
    exited = y_pred >= 0.6

    result = {'attrition_prob': float(y_pred), 'attrition': bool(exited)}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
