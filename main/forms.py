from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField, TextAreaField
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
        
class AdminForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
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

class IndexForm(FlaskForm):
    text = StringField('Input text here')
    submit1 = SubmitField('Analyze text')
    submit2 = SubmitField('Analyze csv')
    file = FileField('Analyze CSV File', validators=[FileAllowed(['csv'])])




class UpdateAccountForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=15)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    picture = FileField('Update Profile Picture', validators=[FileAllowed(['jpg', 'png', 'jpeg'])])
    submit = SubmitField('Update')

    def validate_username(self, username):
        if username.data != current_user.username:
            user = User.query.filter_by(username = username.data).first()
            if user:
                raise ValidationError('This username is taken, please choose a different one')
        

    def validate_email(self, email):
        if email.data != current_user.email:
            user = User.query.filter_by(email = email.data).first()
            if user:
                raise ValidationError('This Email is taken, please choose a different one')

class RequestResetForm(FlaskForm):
        email = StringField('Email',
                        validators=[DataRequired(), Email()])
        submit = SubmitField('Request Password Reset')

        def validate_email(self, email):
            user = User.query.filter_by(email=email.data).first()
            if user is None:
                raise ValidationError('There is no account with that email. You must register first.')


class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')


class LeaveReviewForm(FlaskForm):
    review = TextAreaField('Review', validators=[DataRequired()])
    submit = SubmitField('Submit')

class ElesinObaReviewForm(FlaskForm):
    review = TextAreaField('Review', validators=[DataRequired()])
    submit = SubmitField('Submit')

class TheGhostAndTheToutTooForm(FlaskForm):
    review = TextAreaField('Review', validators=[DataRequired()])
    submit = SubmitField('Submit')

class CitationForm(FlaskForm):
    review = TextAreaField('Review', validators=[DataRequired()])
    submit = SubmitField('Submit')