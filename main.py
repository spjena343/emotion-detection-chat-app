from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('train.txt', delimiter=';', names=['message', 'emotion'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

app = Flask(__name__)
app.secret_key = 'ww'  # Change this to a secure secret key
socketio = SocketIO(app)

messages = []  # Store chat messages
users = {}  # Store user credentials

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if 'signup' in request.form:
            if username not in users:
                users[username] = password
                session['username'] = username
                return redirect(url_for('chat'))
            else:
                return render_template('index.html', error='Username already exists.')
        elif 'login' in request.form:
            if username in users and users[username] == password:
                session['username'] = username
                return redirect(url_for('chat'))
            else:
                return render_template('index.html', error='Invalid username or password.')
    return render_template('index.html', error='')

@app.route('/chat')
def chat():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('chat.html', username=session['username'], messages=messages)


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))



@socketio.on('message')
def handle_message(message):
    if message == "User connected":
        print("User connected")
        message = {
            'username': session['username'],
            'text': message
        }
    else:
        input_vector = vectorizer.transform([message])
        predicted_emotion = model.predict(input_vector)[0]
        print("Message: ", message, " | Emotion: ", predicted_emotion)
        message = {
            'username': session['username'],
            'text': f"[{predicted_emotion}]   {message}"
        }

    messages.append(message)
    emit('message', message, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
