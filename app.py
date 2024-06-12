from flask import Flask, jsonify, make_response, request, abort
import pandas as pd
import catboost
import pickle
from flask_cors import CORS, cross_origin
from waitress import serve

# Load the model
model = pickle.load(open("finalised_model1.sav", "rb"))

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/get_prediction", methods=['POST', 'OPTIONS'])
@cross_origin()
def get_prediction():
    if not request.json:
        abort(400)
    
    # Convert the incoming JSON request to a DataFrame
    df = pd.DataFrame(request.json, index=[0])
    
    # Specify the columns expected by the model
    cols = ["CONSOLE", "RATING", "CRITICS_POINTS", "CATEGORY", "YEAR", "PUBLISHER", "USER_POINTS"]
    df = df[cols]
    
    # Specify which columns are categorical
    categorical_features = ["CONSOLE", "RATING", "CATEGORY", "PUBLISHER"]
    
    # Create a Pool object for CatBoost with categorical features specified
    pool = catboost.Pool(data=df, cat_features=categorical_features)
    
    # Predict using the model
    prediction = model.predict(pool)[0]
    
    return jsonify({'result': prediction}), 201

mode ="prod"

if __name__ == "__main__":
    if mode=="dev":
        app.run(host='0.0.0.0', port=50100, debug=True)
    else:
        serve(app, host='0.0.0.0', port=50100, threads=1)
   
