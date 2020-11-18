"""
This script runs the HelloWorld application using a development server.
"""

from os import environ
from HelloWorld import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        #PORT = int(environ.get('SERVER_PORT', '5555'))
        PORT = 787878
    except ValueError:
        PORT = 5555
    # app.run(HOST, PORT)
    app.run(host="0.0.0.0")
