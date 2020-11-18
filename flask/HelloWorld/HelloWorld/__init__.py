"""
The flask application package.
"""

from flask import Flask
# from flask_restful import reqparse , abort ,Api , Resource 
# api = Api(app)
app = Flask(__name__)
# api = Api(app)

import HelloWorld.views
