from flask import Flask
from flask import request, jsonify, redirect
import pytest
import json
import os
import sys
import flask.json as json
import sample_iris_model_pract03 as iris
from flask import request

app = Flask(__name__)

@app.route('/')
def root():
    python_version = "\npython-version%s\n" % sys.version
    
    return python_version 

@app.route('/iris', methods=['get','post'])
def iris_predict():
    # default result in case error occurs
    result = 'nil'
    # catch errors
    try:
        # get the 4 parameters needed for prediction
        slen = float(request.values.get('slen'))
        sw = float(request.values.get('sw'))
        plen = float(request.values.get('plen'))
        pw = float(request.values.get('pw'))
        # use the trained model to predict
        result = iris.predict_iris(slen, sw, plen, pw)
    except Exception as e:
        print('error:', e) 
    # return result as output
    return result

@app.route("/dialogflow", methods=['post']) # dialogflow uses post method
def dialogflow():
    # default result in case error occurs
    result = {'fulfillmentText':'n/a'}
    try:
        # convert request body from json to python dict
        input = request.get_json()
        # check if correct intent
        if input['queryResult']['intent']['displayName'] == 'iris-predict':
            # get inputs/parameters from the request body as features for model
            params = input['queryResult']['parameters'];
            # sepal_length, sepal_width, petal_length, petal_width
            sepal_length=float(params['sepal_length']) 
            sepal_width=float(params['sepal_width']) 
            petal_length=float(params['petal_length']) 
            petal_width=float(params['petal_width'])

            print((sepal_length, sepal_width, petal_length, petal_width))
            
            # get prediction from trained model
            y_label = iris.predict_iris(
                sepal_length, sepal_width, petal_length, petal_width)
            
            # prepare the predicted label
            output = 'The prediction is {}'.format(y_label)
            # prepare result object
            result = {'fulfillmentText':output}
        else:
            # else, return the input parameters as json
            result = {'fulfillmentText':json.dumps(input['queryResult']['parameters'])}
    except Exception as e:
        print('error:', e)

    # return the json string 
    return json.dumps(result)


if __name__ == '__main__':
    # Get port from environment variable or choose 9099 as local default
    port = int(os.getenv("PORT", 9099))
    # Run the app, listening on all IPs with our chosen port number
    app.run(host='0.0.0.0', port=port, debug=True)


