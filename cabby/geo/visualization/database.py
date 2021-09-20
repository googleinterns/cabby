
from datetime import datetime
from flask import *
from flask_sqlalchemy import SQLAlchemy
from geoalchemy2 import Geometry



db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False    
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
    app.config['SESSION_TYPE'] = 'filesystem'

    db.init_app(app)
    
    return app

class Goal(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  instruction_id = db.Column(db.Integer, nullable=False)
  date_start = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
  date_finish = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
  goal_lat = db.Column(db.Integer, nullable=False)
  goal_lng = db.Column(db.Integer, nullable=False)

  def __repr__(self):
    return f"Goal('{self.id}', '{self.instruction_id}', '{self.goal_lat}', '{self.goal_lng}')"



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
    return "Path Descriptions: \n" + \
      f"id: {self.id}, \n" + \
      f"time: {self.date_start} - {self.date_finish}, \n" + \
      f"rvs_sample_number: {self.rvs_sample_number} \n \n" 