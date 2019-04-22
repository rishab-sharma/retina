from flask import *

app = Flask(__name__)


@app.route('/')
def hello():
    return "<p style='text-align:center'>Hello !! <br>Docker is Done!</p>"
