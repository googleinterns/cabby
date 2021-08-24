from datetime import datetime
from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
import folium
import os
import random

from forms import NavigationForm
import visualize
from database import Instruction, Goal, db, create_app


app = Flask(__name__)


app = create_app()
app.app_context().push()


db.init_app(app)


with app.app_context():
    db.create_all()

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
rvs_path = os.path.join(parent_dir, "pathData/manhattan_geo_paths.gpkg")
osm_maps_instructions = visualize.get_maps_and_instructions(rvs_path)
size_dataset = len(osm_maps_instructions)

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
    sample_number = random.randint(0, size_dataset)
    return redirect(url_for('task', sample=sample_number))
  else:
    session.pop('_flashes', None)
    return render_template('end.html', bar=100)


@app.route("/task/<sample>", methods=['GET', 'POST'])
def task(sample):
  sample = int(sample)
  folium_map, instruction, landmarks, entity = osm_maps_instructions[sample]

  folium_map.save('templates/map.html')
  form = NavigationForm()
  form.landmarks = landmarks

  if form.validate_on_submit():
    content = request.form['content']
    goal = entity.geo_landmarks['end_point'].geometry
    start = entity.geo_landmarks['start_point'].geometry
    path = Instruction(
      rvs_sample_number=sample, 
      content=content, 
      date_finish=datetime.utcnow(),
      rvs_path = rvs_path,
      rvs_goal = str(goal),
      rvs_start = str(start))

    db.session.add(path)
    db.session.commit()

    flash(session["task"], 'success')
    return redirect(url_for('index'))

  if session["task"] == 0:
    task = 0
  else:
    task = session["task"] - 1
  progress_task = round(task / N_TASKS_PER_USER * 100)
  return render_template('instructor_task.html',
                         form=form,
                         bar=progress_task,
                         title=session["task"],
                         )


@app.route('/map')
def map():
  return render_template('map.html')


if __name__ == '__main__':
  sess = Session()
  app.run(host='0.0.0.0', port=5000, debug=False)
