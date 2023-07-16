from datetime import datetime
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from main import db, login_manager, app, admin
from flask_login import UserMixin
from flask_admin.contrib.sqla import ModelView


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True, nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    #phone = db.Column(db.Integer, unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    reviews = db.relationship('Leavereview', backref='user', lazy=True)
    elesinoba_review = db.relationship('ElesinObaReview', backref='user', lazy=True)
    theghostandthetouttoo_review = db.relationship('TheGhostAndTheToutTooReview', backref='user', lazy=True)
    citation_review = db.relationship('CitationReview', backref='user', lazy=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    confirmed = db.Column(db.Boolean, default=False)

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"
    
class Leavereview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text, nullable=False)
    review_sentiment = db.Column(db.Text)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(leave_review):
        return f"Review('{leave_review.review}'), '{leave_review.date_posted}')"
    
class ElesinObaReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text, nullable=False)
    review_sentiment = db.Column(db.Text)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(elesin_oba_review):
        return f"Review('{elesin_oba_review.review}'), '{elesin_oba_review.date_posted}')"
    

class TheGhostAndTheToutTooReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text, nullable=False)
    review_sentiment = db.Column(db.Text)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(theghostandthetouttoo_review):
        return f"Review('{theghostandthetouttoo_review.review}'), '{theghostandthetouttoo_review.review_sentiment}', '{theghostandthetouttoo_review.date_posted}')"
    

class CitationReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text, nullable=False)
    review_sentiment = db.Column(db.Text)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(citation_review):
        return f"Review('{citation_review.review}'), '{citation_review.date_posted}')"