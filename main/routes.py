

import os
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, send_file
from main import app, db, bcrypt, mail
from main.forms import (RegistrationForm, LoginForm, ResetPasswordForm, RequestResetForm)
from main.forms import RegistrationForm, LoginForm, UpdateAccountForm, PostForm
from main.models import User
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
import subprocess


import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from io import BytesIO
import chardet



@app.route("/")
@app.route("/home")
def home_page():
    return render_template('homepage.html')

@app.route("/about/")
def about():
    return render_template('about.html', title='about')



@app.route("/registration", methods=['GET', 'POST'])
def registration():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created. you can now login','success')
        return redirect(url_for('login'))
    return render_template('registration.html', title='Registration', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home_page'))
        else:
            flash('Login Unsuccessful. Please check email or password', 'danger')
    return render_template('login.html', title='Login', form=form)



@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home_page'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profilepics', picture_fn)
    
    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form= UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Account Updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profilepics/' + current_user.image_file)
    return render_template('account.html', title='Account', image_file=image_file, form=form)


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='mylesadebayo@gmail.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)

 
@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)





# Load the dataset
df = pd.read_csv('sentiment.csv')

# Preprocess the data
nltk.download('punkt')
nltk.download('stopwords')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['yo_mt_review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the sentiment analysis model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy:', accuracy)

# Save the model
joblib.dump(model, 'sentiment_analysis_model.joblib')

# Load the sentiment analysis model
loaded_model = joblib.load('sentiment_analysis_model.joblib')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = loaded_model.predict(vectorizer.transform([text]))[0]
        return render_template('index.html', text=text, sentiment=sentiment)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        content = file.read()
        result = chardet.detect(content)
        encoding = result['encoding']

        try:
            df = pd.read_csv(BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(content), encoding='latin-1')

        column_heading = df.columns[0]  # Get the first column heading

        df['analysis'] = df[column_heading].apply(lambda x: loaded_model.predict(vectorizer.transform([x]))[0])

        # Get the first 10 rows for display
        table_data = df.head(10).to_html(index=False)

        return render_template('index.html', table_data=table_data)



#trial
@app.route('/post/new', methods=['GET','POST'])
@login_required
def new_post():
    form =PostForm()
    if form.validate_on_submit():
         flash
    return render_template('create_post.html', title='New Post')
