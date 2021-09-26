# app.py
# Required Imports
import os
from flask import Flask, request, jsonify
from flask import *

# from firebase_admin import credentials, firestore, initialize_app
# Initialize Flask App
app = Flask(__name__)
# Initialize Firestore DB
# cred = credentials.Certificate('/home/onlp_gcp_biu/key.json')
# default_app = initialize_app(cred)
# db = firestore.client()
# todo_ref = db.collection('todos')


@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')


port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port)