from flask import Flask , session , render_template , request , redirect ,url_for, g
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/', methods=['GET','POST'])
def index() :
    if request.method == 'POST' :

        session.pop('user' , None) #이케 계속 지우면 로그인은 한명바계못하지않나?
        # 아이디 비밀번호 체크.
        if request.form['password'] == 'password':
            session['user'] = request.form['username']
            return redirect(url_for('protected'))
    return render_template('nonprotect.html')

@app.route('/getsession')
def getsession() :
    if 'user' in session:
        return session['user']
    
    return 'Not logged in!'


@app.before_request 
def before_request() :
    g.user = None 
    if 'user' in session :
        g.user = session['user']

@app.route('/protected')
def protected():
    if g.user:
        return render_template('protect.html')

    return redirect(url_for('index'))


@app.route('/dropsession')
def dropsession():
    session.pop('user',None)  # 뒤에 none은 replacing it with something을 안하겟다
    return 'Drop'
    # drop 하기 전까지는 계속 유지되네


if __name__ =='__main__' :
    app.run(debug=True)


#There are multiple ways of doing session management is flask
# 이건 simplest way

# 세션다루는건 return 같은거없이 플라스크 내부에서만하니까 뭔가 리액트와 연동도 잘되려나

# g가 스레드 머라는데 잘이해안됨 한번더보기
# https://www.youtube.com/watch?v=eBwhBrNbrNI&index=3&list=PLXmMXHVSvS-CMpHUeyIeqzs3kl-tIG-8R 