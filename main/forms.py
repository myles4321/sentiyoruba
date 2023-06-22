from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from main.models import User

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=15)])
    email = StringField('Email', validators=[DataRequired(), Email()])
   # phone = IntegerField('Phone Number', validators=[DataRequired(), Length(min=9, max=9)] )
    password = PasswordField('Password', validators=[DataRequired(), Length(min=5)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])


    submit = SubmitField('Sign Up')

    def validate_username(self, username):

        user = User.query.filter_by(username = username.data).first()
        if user:
            raise ValidationError('This username is taken, please choose a different one')
        

    def validate_email(self, email):

        user = User.query.filter_by(email = email.data).first()
        if user:
            raise ValidationError('This Email is taken, please choose a different one')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=5)])
    remember = BooleanField('Remember Me')


    submit = SubmitField('Login')