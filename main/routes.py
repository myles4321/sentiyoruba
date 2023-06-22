
from flask import render_template, url_for, flash, redirect, request
from main import app, db, bcrypt
from main.forms import RegistrationForm, LoginForm
from main.models import User
from flask_login import login_user, current_user, logout_user, login_required

posts= [
    {
        'author' : 'Myles Johnson',
        'title' : 'Blog post',
        'content' : 'How to be a good boy',
        'date_posted' : 'March 17, 2023'
    },
      {
        'author' : 'Johns doe',
        'title' : 'Blog review',
        'content' : 'How the world works',
        'date_posted' : 'April 5, 2025'
    }
]

@app.route("/")
@app.route("/home")
def home_page():
    return render_template('homepage.html', blog=posts)

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



@app.route("/account")
@login_required
def account():
    return render_template('account.html', title='Account')

