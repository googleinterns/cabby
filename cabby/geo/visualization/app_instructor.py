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
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

from forms import NavigationForm, ReviewForm
import visualize
from database import Instruction, Goal, db, create_app


# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
instructions_ref = db.collection('instructions')
instructions_ref_sandbox = db.collection('instructions_sandbox')

app = Flask(__name__)

app = create_app()
SECRET_KEY = '5791628bb0b13ce0c676dfde280ba245'
app.config['SECRET_KEY'] = SECRET_KEY

app.app_context().push()

try:
  rvs_path = os.path.abspath("./data/manhattan_samples_v4.gpkg")
except Exception as e:
  print (f"An Error Occured: {e}, {rvs_path}")

osm_maps_instructions = visualize.get_maps_and_instructions(rvs_path)
size_dataset = len(osm_maps_instructions)

N_TASKS_PER_USER = 2

path_map = os.path.join(app.root_path,"templates", secure_filename('map.html'))
if os.path.exists(path_map):
    os.remove(path_map)


task=1
sample = random.randint(0, size_dataset)
date_start = datetime.utcnow()


@app.route("/", methods=['GET', 'POST'])
@app.route("/pub/", methods=['GET', 'POST'])
def home():
  global sample
  global date_start


  assignmentId = request.args.get("assignmentId") 
  assignmentId = assignmentId if assignmentId else uuid.uuid4().hex
  hitId = request.args.get("hitId")
  hitId = hitId if hitId else uuid.uuid4().hex
  turkSubmitTo = request.args.get("turkSubmitTo")
  workerId = request.args.get("workerId")
  workerId = workerId if workerId else uuid.uuid4().hex

  turkSubmitTo = request.args.get("turkSubmitTo")
  turkSubmitTo = turkSubmitTo if turkSubmitTo else "https://workersandbox.mturk.com"
  
  
  folium_map, instruction, landmarks, entity = osm_maps_instructions[sample]

  if os.path.exists(path_map):
    os.remove(path_map)

  folium_map.save(path_map)

  form_nav = NavigationForm()

  form_nav.landmarks = landmarks
  
  if request.method == 'POST': 
    if os.path.exists(path_map):
      os.remove(path_map)
    folium_map.save(path_map)
    
    if form_nav.validate_on_submit() or (
      len(form_nav.errors)==1 and 'csrf_token' in form_nav.errors):

      # session['task'] = task
      content = request.form['content']
      goal_point = entity.geo_landmarks['end_point'].geometry
      start_point = entity.geo_landmarks['start_point'].geometry

      sample = random.randint(0, size_dataset)


      try:
        j_req = {
        'hit_id': hitId,
        'work_id': workerId,
        'assignmentId': assignmentId, 
        'rvs_sample_number': str(sample),
        'content': content,
        'rvs_path': rvs_path,
        'rvs_goal_point': str(goal_point),
        'rvs_start_point': str(start_point),
        'date_start': str(date_start),
        'date_finish': str(datetime.utcnow())}

        # save to database
        id = workerId + hitId + str(task)
        if 'sandbox' in turkSubmitTo:
          instructions_ref_sandbox.document(id).set(j_req)
        else:
          instructions_ref.document(id).set(j_req)
      except Exception as e:
        return f"An Error Occured: {e}"


      address = turkSubmitTo + '/mturk/externalSubmit'
      fullUrl = address + '?assignmentId=' + assignmentId + '&workerId=' + workerId + "&hitId=" + hitId
      return render_template(
        'end.html', 
        bar=100, 
        fullUrl=fullUrl,
      ) 

  date_start = datetime.utcnow()

  if task == 0:
    task_bar = 0
  else:
    task_bar = task - 1
  progress_task = round(task_bar / N_TASKS_PER_USER * 100)
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
  return render_template("404.html", exc = e)

@app.route('/map')
def map():
  return render_template('map.html')

port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':

  app.run(threaded=True, host='0.0.0.0', port=port, debug=True)
