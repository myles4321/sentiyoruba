
import os
import secrets
import tempfile
from PIL import Image
from flask import render_template, sessions, url_for, flash, redirect, request, send_file, abort, make_response, session
from itsdangerous import BadSignature, Serializer, TimedSerializer, URLSafeTimedSerializer
from yaml import serialize_all 
from main import app, db, bcrypt, mail, admin
from main.forms import (RegistrationForm, LoginForm, ResetPasswordForm, RequestResetForm, IndexForm, LeaveReviewForm, ElesinObaReviewForm, TheGhostAndTheToutTooForm, CitationForm, AdminForm)
from main.forms import RegistrationForm, LoginForm, UpdateAccountForm
from main.models import User, Leavereview, ElesinObaReview, TheGhostAndTheToutTooReview, CitationReview
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message, Mail
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from io import BytesIO
import chardet
from flask_admin.contrib.sqla import ModelView



@app.route("/")
@app.route("/home")
def home_page():
    return render_template('homepage.html')

@app.route("/about/")
def about():
    return render_template('about.html', title='about')


mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

@app.route("/registration", methods=['GET', 'POST'])
def registration():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)

        # Generate a confirmation token
        token = serializer.dumps(user.email, salt='email-confirm')

        # Send confirmation email
        confirmation_link = url_for('confirm_email', token=token, _external=True)
        message = Message('Confirm Your Email', recipients=[user.email], sender=app.config['MAIL_USERNAME'])
        message.body = f'Please click the link to confirm your email: {confirmation_link}'
        mail.send(message)

        db.session.add(user)
        db.session.commit()
        flash('Your account has been created. Please check your email to confirm your account.', 'success')
        return redirect(url_for('login'))
    return render_template('registration.html', title='Registration', form=form)

@app.route('/confirm_email/<token>')
def confirm_email(token):
    try:
        email = serializer.loads(token, salt='email-confirm', max_age=1800)  # 30 minutes expiration

        user = User.query.filter_by(email=email).first()
        if user:
            user.confirmed = True
            db.session.commit()
            flash('Email confirmed. You can now log in.', 'success')
        else:
            flash('User not found.', 'danger')

    except BadSignature:
        flash('The confirmation link is invalid or has expired.', 'danger')

    return redirect(url_for('login'))

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            if user.confirmed:
                login_user(user, remember=form.remember.data)
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('home_page'))
            else:
                flash('Please confirm your email address to log in.', 'warning')
        else:
            flash('Login Unsuccessful. Please check email or password.', 'danger')
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

# Define the Yoruba stopwords
yoruba_stopwords = ["a","an","bá","bí","bẹ̀rẹ̀","fún","fẹ́","gbogbo","inú","jù","jẹ","jẹ́","kan","kì","kí","kò","láti","lè","lọ","mi","mo","máa","mọ̀","ni","náà","ní",
                    "nígbà","nítorí","nǹkan","o","padà","pé","púpọ̀","pẹ̀lú","rẹ̀","sì","sí","sínú","ṣ","ti","tí","wà","wá","wọn","wọ́n","yìí","àti","àwọn","é","í",
                    "òun","ó","ń","ńlá","ṣe","ṣé","ṣùgbọ́n","ẹmọ́","ọjọ́","ọ̀pọ̀lọpọ̀"]

# Define the tokenizer with custom Yoruba stopwords
def tokenizer(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in yoruba_stopwords]
    return filtered_tokens

df['filtered_tokens'] = df['yo_review'].apply(tokenizer)

# Vectorize the text data
vectorizer = TfidfVectorizer(tokenizer=tokenizer)
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


@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    form = IndexForm()
    text = None
    sentiment = None
    tokenized_text = None
    table_data = None
    df = None

    if request.method == 'POST' and form.validate_on_submit():
        text = form.text.data
        tokenized_text = tokenizer(text)
        sentiment = loaded_model.predict(vectorizer.transform([text]))[0]

        file = request.files['file']

        if file.filename == '':
            flash('No file selected.')
        elif file:
            content = file.read()

            if len(content) == 0:
                flash('Empty file.')
            else:
                result = chardet.detect(content)
                encoding = result['encoding']

                try:
                    df = pd.read_csv(BytesIO(content), encoding=encoding)
                    column_heading = df.columns[0]  # Get the first column heading

                    df['analysis'] = df[column_heading].apply(lambda x: loaded_model.predict(vectorizer.transform([x]))[0])
                    
                    session['submitted_data'] = df.to_csv(index=False)

                    # Get the first 10 rows for display
                    table_data = df.head(10).to_html(index=False)

                except UnicodeDecodeError:
                    flash('Unable to decode file with the provided encoding.')
                except pd.errors.EmptyDataError:
                    flash('File contains no data or invalid format.')

    return render_template('index.html', form=form, text=text, sentiment=sentiment, tokenized_text=tokenized_text, table_data=table_data, df=df)




# ...


@app.route('/download_data')
@login_required
def download_data():
    submitted_data = session.get('submitted_data')

    if not submitted_data:
        # Handle the case where no data was submitted or provide an appropriate response
        return "No data was submitted for download."

    # Set the appropriate filename for the download
    filename = 'full_data.csv'

    # Create the response object
    response = make_response(submitted_data)
    response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    response.headers['Content-Type'] = 'text/csv'

    return response






#reviews routes

@app.route("/view_reviews1")
@login_required
def view_reviews1():
    leave_review = Leavereview.query.all()
    total_reviews = len(leave_review)
    positive_reviews = 0

    if total_reviews > 0:
        for review in leave_review:
            sentiment = loaded_model.predict(vectorizer.transform([review.review]))[0]
            review.review_sentiment = sentiment
            if sentiment == 'positive':
                positive_reviews += 1
        negative_reviews = total_reviews - positive_reviews
        percentage_positive = (positive_reviews / total_reviews) * 100
        percentage_negative = (negative_reviews / total_reviews) * 100

        db.session.commit()
        return render_template('view_reviews1.html', review=leave_review, total_reviews=total_reviews, percentage_positive=percentage_positive, percentage_negative=percentage_negative)
    else:
        return render_template('view_reviews1.html', total_reviews=total_reviews)


@app.route("/view_reviews2")
@login_required
def view_reviews2():
    obaelesin_review = ElesinObaReview.query.all()
    total_reviews = len(obaelesin_review)
    positive_reviews = 0

    if total_reviews > 0:
        for review in obaelesin_review:
            sentiment = loaded_model.predict(vectorizer.transform([review.review]))[0]
            review.review_sentiment = sentiment
            if sentiment == 'positive':
                positive_reviews += 1
        negative_reviews = total_reviews - positive_reviews
        percentage_positive = (positive_reviews / total_reviews) * 100
        percentage_negative = (negative_reviews / total_reviews) * 100

        db.session.commit()
        return render_template('view_reviews2.html', review=obaelesin_review, total_reviews=total_reviews, percentage_positive=percentage_positive, percentage_negative=percentage_negative)
    else:
        return render_template('view_reviews2.html', total_reviews=total_reviews)

@app.route("/view_reviews3")
@login_required
def view_reviews3():
    theghostandthetouttoo_Review = TheGhostAndTheToutTooReview.query.all()
    total_reviews = len(theghostandthetouttoo_Review)
    positive_reviews = 0

    if total_reviews > 0:
        for review in theghostandthetouttoo_Review:
            sentiment = loaded_model.predict(vectorizer.transform([review.review]))[0]
            review.review_sentiment = sentiment
            if sentiment == 'positive':
                positive_reviews += 1
        negative_reviews = total_reviews - positive_reviews
        percentage_positive = (positive_reviews / total_reviews) * 100
        percentage_negative = (negative_reviews / total_reviews) * 100

        db.session.commit()
        return render_template('view_reviews3.html', review=theghostandthetouttoo_Review, total_reviews=total_reviews, percentage_positive=percentage_positive, percentage_negative=percentage_negative)
    else:
        return render_template('view_reviews3.html', total_reviews=total_reviews)



@app.route("/view_reviews4")
@login_required
def view_reviews4():
    citation_review = CitationReview.query.all()
    total_reviews = len(citation_review)
    positive_reviews = 0

    if total_reviews > 0:
        for review in citation_review:
            sentiment = loaded_model.predict(vectorizer.transform([review.review]))[0]
            review.review_sentiment = sentiment
            if sentiment == 'positive':
                positive_reviews += 1
        negative_reviews = total_reviews - positive_reviews
        percentage_positive = (positive_reviews / total_reviews) * 100
        percentage_negative = (negative_reviews / total_reviews) * 100

        db.session.commit()
        return render_template('view_reviews4.html', review=citation_review, total_reviews=total_reviews, percentage_positive=percentage_positive, percentage_negative=percentage_negative)
    else:
        return render_template('view_reviews4.html', total_reviews=total_reviews)


#leave a review routes

@app.route("/leave_a_review1", methods=['GET', 'POST'])
@login_required
def leave_a_review1():
    form = LeaveReviewForm()
    if form.validate_on_submit():
        review = Leavereview(review=form.review.data, user=current_user)
        db.session.add(review)
        db.session.commit()
        flash('Your review has been sent!', 'success')
        return redirect(url_for('home_page'))
    return render_template('leave_a_review1.html', title='New Review', form=form)

@app.route("/leave_a_review2", methods=['GET', 'POST'])
@login_required
def leave_a_review2():
    form = ElesinObaReviewForm()
    if form.validate_on_submit():
        review = ElesinObaReview(review=form.review.data, user=current_user)
        db.session.add(review)
        db.session.commit()
        flash('Your review has been sent!', 'success')
        return redirect(url_for('home_page'))
    return render_template('leave_a_review2.html', title='New Review', form=form)

@app.route("/leave_a_review3", methods=['GET', 'POST'])
@login_required
def leave_a_review3():
    form = TheGhostAndTheToutTooForm()
    if form.validate_on_submit():
        review = TheGhostAndTheToutTooReview(review=form.review.data, user=current_user)
        db.session.add(review)
        db.session.commit()
        flash('Your review has been sent!', 'success')
        return redirect(url_for('home_page'))
    return render_template('leave_a_review3.html', title='New Review', form=form)

@app.route("/leave_a_review4", methods=['GET', 'POST'])
@login_required
def leave_a_review4():
    form = CitationForm()
    if form.validate_on_submit():
        review = CitationReview(review=form.review.data, user=current_user)
        db.session.add(review)
        db.session.commit()
        flash('Your review has been sent!', 'success')
        return redirect(url_for('home_page'))
    return render_template('leave_a_review4.html', title='New Review', form=form)



class Controller(ModelView):
    def is_accessible(self):
        if current_user.is_admin == True:
            return current_user.is_authenticated
        else:
            return abort(404)
    def not_auth(self):
        return "You are not allowed to view this page"
    
admin.add_view(Controller(User, db.session))
admin.add_view(Controller(Leavereview, db.session))
admin.add_view(Controller(ElesinObaReview, db.session))
admin.add_view(Controller(TheGhostAndTheToutTooReview, db.session))
admin.add_view(Controller(CitationReview, db.session))

@app.route('/172002', methods=['GET', 'POST'])
def admin_signup():
    if current_user.is_authenticated:
        return redirect(url_for('home_page'))
    form = AdminForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(email=form.email.data, password=hashed_password, is_admin=True)
        db.session.add(user)
        db.session.commit()
        flash('Account Created Successfully. Login with your details!','success')
        return redirect(url_for('login'))
    return render_template('admin-signup.html', title = 'Sign up', form = form)