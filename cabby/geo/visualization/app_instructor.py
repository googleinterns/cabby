from datetime import datetime
import flask
from flask import *
from flask import session
from flask_sqlalchemy import SQLAlchemy
import folium
import os
import random
import uuid
from firebase_admin import credentials, firestore, initialize_app


from forms import NavigationForm
import visualize
from database import Instruction, Goal, db, create_app


# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
todo_ref = db.collection('todos')

app = Flask(__name__)

app = create_app()
SECRET_KEY = '5791628bb0b13ce0c676dfde280ba245'
app.config['SECRET_KEY'] = SECRET_KEY

app.app_context().push()



# db.init_app(app)


# with app.app_context():
#     db.create_all()

rvs_path = os.path.abspath("./data/manhattan_geo_paths.gpkg")
osm_maps_instructions = visualize.get_maps_and_instructions(rvs_path)
size_dataset = len(osm_maps_instructions)

N_TASKS_PER_USER = 2



@app.route("/")
@app.route("/pub/")
def home():
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
  form_nav = NavigationForm()

  form_nav.landmarks = landmarks
  date_start = datetime.utcnow()

  if request.method == 'POST':
    
    if form_nav.validate_on_submit() or (
      len(form_nav.errors)==1 and 'csrf_token' in form_nav.errors):

      session['task'] = task
      content = request.form['content']
      goal_point = entity.geo_landmarks['end_point'].geometry
      start_point = entity.geo_landmarks['start_point'].geometry
    

      try:
        j_req = {
        'hit_id': '1',
        'work_id': 's', 
        'rvs_sample_number': str(task),
        'content': content,
        'rvs_path': rvs_path,
        'rvs_goal_point': str(goal_point),
        'rvs_start_point': str(start_point),
        'date_start': str(date_start),
        'date_finish': str(datetime.utcnow())}

        # j_req={'id': '1', 'title': 'Write a blog post'}

        id = uuid.uuid4().hex
        todo_ref.document(id).set(j_req)
      except Exception as e:
        return f"An Error Occured: {e}"

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
                         form=form_nav,
                         bar=progress_task,
                         title=task,
                         )



@app.errorhandler(500)
def internal_server_error(e):
  return jsonify(error=str(e)), 500

@app.errorhandler(404)
def page_not_found(e):
  return render_template("404.html")

@app.route('/map')
def map():
  return render_template('map.html')

port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':

  app.run(threaded=True, host='0.0.0.0', port=port, debug=True)
