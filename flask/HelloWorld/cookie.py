from flask import Flask , make_response , request

app = Flask(__name__)

@app.route("/set")
def setcookie() :
    resp = make_response('Setting Cookie!')
    resp.set_cookie('framework' , 'flask') # key value
    # make_response로  return whatever you want . 
    return resp

@app.route("/get")
def getcookie():
    framework = request.cookies.get('framework')

    # any form , json , cookie 다 잡아 읽을수있는듯
    return 'The framework is ' + framework

if __name__ =='__main__' :
    app.run(debug=True)


