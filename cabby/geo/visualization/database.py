
from datetime import datetime
from flask import *
from flask_sqlalchemy import SQLAlchemy
from geoalchemy2 import Geometry

from sqlalchemy.ext.declarative import declarative_base


db = SQLAlchemy()

Base = declarative_base()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False    
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
    app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
    app.config['SESSION_TYPE'] = 'filesystem'
    db.init_app(app)
    
    return app

class Goal(Base):
  __tablename__ = 'goal'
  id = db.Column(db.Integer, primary_key=True)
  instruction_id = db.Column(db.Integer, nullable=False)
  date_start = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
  date_finish = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
  goal_point = db.Column(Geometry(geometry_type='POLYGON', management=True))

  def __repr__(self):
    return f"Goal('{self.id}', '{self.sample_number}', '{self.date_finish}')"



class Instruction(db.Model):
  
  id = db.Column(db.Integer, primary_key=True)
  rvs_sample_number = db.Column(db.Integer, nullable=False)
  content = db.Column(db.Text, nullable=False)
  date_start = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
  date_finish = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
  rvs_path = db.Column(db.Text, nullable=False)
  rvs_goal = db.Column(db.Text, nullable=False)
  rvs_start = db.Column(db.Text, nullable=False)


  def __repr__(self):
    return f"Path Descriptions('{self.id}', '{self.rvs_sample_number}', '{self.date_finish}', '{self.rvs_path}')"

