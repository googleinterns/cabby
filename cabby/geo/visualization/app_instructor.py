from datetime import datetime
import flask
from flask import *
from flask import session
from flask_sqlalchemy import SQLAlchemy
import folium
import os
import random
from OpenSSL import SSL
from jinja2.utils import contextfunction
from OpenSSL import crypto
# from flask_wtf import CSRFProtect
from flask_wtf import csrf
# from flask_wtf.csrf import CSRFProtect
from flask_wtf.csrf import CSRFProtect
import psiturk 
import uuid

from forms import NavigationForm
import visualize
from database import Instruction, Goal, db, create_app


app = Flask(__name__)
csrf = CSRFProtect()
csrf.init_app(app)


app = create_app()
SECRET_KEY = '5791628bb0b13ce0c676dfde280ba245'
app.config['SECRET_KEY'] = SECRET_KEY

app.app_context().push()



db.init_app(app)


with app.app_context():
    db.create_all()

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
rvs_path = os.path.join(parent_dir, "pathData/manhattan_geo_paths.gpkg")
osm_maps_instructions = visualize.get_maps_and_instructions(rvs_path)
size_dataset = len(osm_maps_instructions)

N_TASKS_PER_USER = 2


@app.route("/")
@app.route("/pub")
def home():
  print ("???????????????????????????????????????????????????????????????")
  task = 0
  
  return redirect(url_for(
    'index', 
    task=task,
    ))
    


@app.route("/index/<task>", methods=['GET', 'POST'])
def index(task):
  task = int(task)

  task +=1
  if task <= N_TASKS_PER_USER:
    sample_number = random.randint(0, size_dataset)
    return redirect(url_for(
      'task', 
      sample=sample_number, 
      task=task, 
     ))
  else:
    session.pop('_flashes', None)


    return render_template(
      'end.html', 
      bar=100, 
      code=uuid.uuid4().hex
      )


@app.route("/task/<sample>/<task>", methods=['GET', 'POST'])
def task(sample, task):
  sample = int(sample)
  task = int(task)
  folium_map, instruction, landmarks, entity = osm_maps_instructions[sample]

  folium_map.save('templates/map.html')
  form = NavigationForm(secret_key=SECRET_KEY)

  form.landmarks = landmarks


  print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  print (form.errors)
  print (len(form.errors))

  if flask.request.method == 'POST' and len(form.errors)==0:

    session['task'] = task
    content = request.form['content']
    goal = entity.geo_landmarks['end_point'].geometry
    start = entity.geo_landmarks['start_point'].geometry

    path = Instruction(
        date_start = datetime.utcnow(),
        rvs_sample_number=task, 
        content=content, 
        rvs_path = rvs_path,
        rvs_goal = str(goal),
        rvs_start = str(start))

    db.session.add(path)
    db.session.commit()
    return redirect(url_for(
      'index', 
      task=task, 
      )) 

  if task == 0:
    task = 0
  else:
    task = task - 1
  progress_task = round(task / N_TASKS_PER_USER * 100)
  return render_template('instructor_task.html',
                         form=form,
                         bar=progress_task,
                         title=task,
                         )


@app.route('/map')
def map():
  return render_template('map.html')


if __name__ == '__main__':
  sess = Session()

  app.run()
