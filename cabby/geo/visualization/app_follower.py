from datetime import datetime
from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
import os
import random


from database import Instruction, Goal, db, create_app
import visualize
from .. import util
from .. import walk


app = create_app()
app.app_context().push()


db.init_app(app)



with app.app_context():
    db.create_all()

sess = Session()

size_dataset = db.session.query(Instruction).count()

N_TASKS_PER_USER = 5

@app.route("/")
@app.route("/home")
def home():
  session["task"] = 0
  return redirect(url_for('index'))


@app.route("/index")
def index():
  session["task"] += 1
  if session["task"] <= N_TASKS_PER_USER:
    sample_number = random.randint(0, size_dataset-1)
    session["sample"] = sample_number
    return redirect(url_for('task'))
  else:
    session.pop('_flashes', None)
    return render_template('end.html', bar=100)


@app.route("/task", methods=['GET', 'POST'])
def task():
  sample = int(session["sample"])
  row = db.session.query(Instruction)[sample]
  entities = walk.load_entities(row.rvs_path)
  entity = entities[row.rvs_sample_number]
  start_point = util.list_yx_from_point(entity.geo_landmarks['start_point'].geometry)
  nav_instruction = row.content

  if session["task"] == 0:
    task = 0
  else:
    task = session["task"] - 1

  progress_task = round(task / N_TASKS_PER_USER * 100)

  if request.method == "POST":
          return redirect(url_for('index'))


  return render_template('follower_task.html',
                         start_point=start_point,
                         nav_instruction=nav_instruction,
                         bar=progress_task,
                         title=session["task"],
                         )

@app.route('/map')
def map():
  return render_template('map.html')

@app.route("/button", methods=['GET', 'POST'])
def button():
  sample = int(session["sample"])
  row = db.session.query(Instruction)[sample]
  lat = request.json['lat']
  lng = request.json['lng']
  goal_row = Goal(
    instruction_id=row.id, 
    date_finish=datetime.utcnow(),
    goal_lat =lat,
    goal_lng = lng
    )
  db.session.add(goal_row)
  db.session.commit()
  return redirect(url_for('task'))

if __name__ == '__main__':
  sess = Session()
  app.run(host='0.0.0.0', port=5001, debug=True)
