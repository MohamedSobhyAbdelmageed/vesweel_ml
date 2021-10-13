from flask import Flask, render_template , request , jsonify
import joblib
import numpy as np
# from helpers.dummies import *

app = Flask(__name__)
model = joblib.load('brain/model.h5')
scaler = joblib.load('brain/scaler.h5')
print("model is loading")


# //define routs
@app.route('/',methods=['GET'])
def index():
    LocationNormalized = int(request.args['l1'])
    ContractTime = int(request.args['c1'])
    Category = int(request.args['c2'])
    # LocationNormalized: 
    # 4
    # ContractTime: 
    # 1
    # Category: 
    # 5
    # LocationNormalizedis_missing: 
    # 1
    # ContractTimeis_missing: 
    # 1
    # Categoryis_missing: 
    # 1

    data = [ LocationNormalized , ContractTime , Category , 1 , 1 , 1 ]
    result = round(model.predict(scaler.transform([data]))[0])

    return jsonify(prediction = str(result))

if __name__ == "__main__":
    app.run(debug=True)

