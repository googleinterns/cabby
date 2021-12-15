from datetime import datetime
from datetime import timedelta
import flask
from flask import *
from flask import session
from flask_session import Session
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
import util

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred)
db = firestore.client()
instructions_ref = db.collection('instructions')
instructions_ref_sandbox = db.collection('instructions_sandbox')

verification_ref = db.collection('verification')
verification_ref_sandbox = db.collection('verification_sandbox')

app = Flask(__name__)

app = create_app()
SECRET_KEY = '5791628bb0b13ce0c676dfde280ba245'
app.config['SECRET_KEY'] = SECRET_KEY

app.app_context().push()

Session(app)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)



try:
  rvs_path = os.path.abspath("./data/manhattan_samples_v13.gpkg")
except Exception as e:
  print (f"An Error Occured: {e}, {rvs_path}")

# Raw Data
osm_maps_instructions = visualize.get_maps_and_instructions(rvs_path)
size_dataset = len(osm_maps_instructions)

task_session = {}
sample_session = {}
start_session = {}


N_TASKS_PER_USER = 1

# path_map = os.path.join(app.root_path,"templates", secure_filename('map.html'))
# if os.path.exists(path_map):
#     os.remove(path_map)


# sample = random.randint(0, size_dataset)
# date_start = datetime.utcnow()
dir_map = os.path.join(app.root_path,"templates")

@app.route("/", methods=['GET', 'POST'])
@app.route("/pub/", methods=['GET', 'POST'])
@app.route("/home/", methods=['GET', 'POST'])
def home():
  global sample
  global task_session

  assignmentId = request.args.get("assignmentId") 
  assignmentId = assignmentId if assignmentId else uuid.uuid4().hex
  hitId = request.args.get("hitId")
  hitId = hitId if hitId else "2"
  turkSubmitTo = request.args.get("turkSubmitTo")
  workerId = request.args.get("workerId")
  workerId = workerId if workerId else "1"
  turkSubmitTo = request.args.get("turkSubmitTo")
  turkSubmitTo = turkSubmitTo if turkSubmitTo else "https://workersandbox.mturk.com"


  session_id = workerId+hitId
  if session_id not in task_session:
    task_session[session_id] = 1

  if session_id not in start_session:
    start_session[session_id] = datetime.utcnow()
  
  if workerId not in sample_session:
    sample_session[workerId] = random.randint(0, size_dataset-1)

  if task_session[session_id] % 2!=0:

    return verification_task(
      workerId=workerId, 
      hitId=hitId, 
      assignmentId=assignmentId, 
      session_id=session_id, 
      turkSubmitTo=turkSubmitTo)
  
  return description_task(
      workerId=workerId, 
      hitId=hitId, 
      assignmentId=assignmentId, 
      session_id=session_id, 
      turkSubmitTo=turkSubmitTo)
  

def description_task(
  workerId, hitId, assignmentId, session_id, turkSubmitTo):
  folium_map, instruction, landmarks, entity = osm_maps_instructions[
    sample_session[workerId]]
  path_map = os.path.join(dir_map,f"map_{sample_session[workerId]}.html") 
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

      content = request.form['content']
      goal_point = entity.geo_landmarks['end_point'].geometry
      start_point = entity.geo_landmarks['start_point'].geometry

      try:
        id = workerId + hitId + str(task_session[session_id])

        j_req = {
        'hit_id': hitId,
        'work_id': workerId,
        'assignmentId': assignmentId, 
        'rvs_sample_number': str(sample_session[workerId]),
        'content': content,
        'rvs_path': rvs_path,
        'rvs_goal_point': str(goal_point),
        'rvs_start_point': str(start_point),
        'task': task_session[session_id],
        'date_start': str(start_session[session_id]),
        'date_finish': str(datetime.utcnow()),
        'key': id
        }

        # save to database
        if 'sandbox' in turkSubmitTo:
          instructions_ref_sandbox.document(id).set(j_req)
        else:
          instructions_ref.document(id).set(j_req)
      except Exception as e:
        return f"An Error Occured: {e}"

      
      sample_session[workerId] = random.randint(0, size_dataset-1)

      folium_map, instruction, landmarks, entity = osm_maps_instructions[sample_session[workerId]]
      path_map = os.path.join(dir_map,f"map_{sample_session[workerId]}.html") 
      if os.path.exists(path_map):
        os.remove(path_map)
      folium_map.save(path_map)

      form_nav = NavigationForm()
      form_nav.landmarks = landmarks
      
      task_session[session_id] = task_session[session_id] + 1

      
      if task_session[session_id]>N_TASKS_PER_USER:
        task_session[session_id] = 1
        address = turkSubmitTo + '/mturk/externalSubmit'
        fullUrl = address + '?assignmentId=' + assignmentId + '&workerId=' + workerId + "&hitId=" + hitId
        return render_template(
          'end.html', 
          bar=100, 
          fullUrl=fullUrl,
        )
      else:
        return home()


  start_session[session_id] = datetime.utcnow()

  if task_session[session_id] == 0:
    task_bar = 0
  else:
    task_bar = task_session[session_id] - 1
  progress_task = round(task_bar / N_TASKS_PER_USER * 100)
  
  title = 0 if task_session[session_id]==1 else task_session[session_id]

  return render_template('instructor_task.html',
                         form=form_nav,
                         bar=progress_task,
                         title=title,
                         n_sample = sample_session[workerId]
                         )
                         

def verification_task(
  workerId, turkSubmitTo, session_id, hitId, assignmentId):

  # read from database
  table_name = 'instructions'
  
  instruction_data_all = list(db.collection(table_name).get())

  instruction_data = [
    e.to_dict() for e in instruction_data_all if 'review' in e.to_dict() and e.to_dict()['review']=='RVS_excellent']
  data_len = len(instruction_data)
  n_sample = random.randint(0, data_len-1)

  sample = instruction_data[n_sample]


  path_data = os.path.abspath(sample['rvs_path'].replace("/app_instructor", "."))

  entities = util.load_entities(path_data)

  try:
    entity = entities[int(sample['rvs_sample_number'])]
  except:
    print (sample['rvs_sample_number'], len(entities))

  start_point = util.list_yx_from_point(entity.geo_landmarks['start_point'].geometry)
  nav_instruction = sample['content']

  if task_session[session_id] == 0:
    task_bar = 0
  else:
    task_bar = task_session[session_id] - 1

  progress_task = round(task_bar / N_TASKS_PER_USER * 100)

  title = 0 if task_session[session_id]==1 else task_session[session_id]
  

  form = ReviewForm()

  if request.method == 'POST': 

    
    if request.form.get("submit_button"):
      
      latlng_dict = json.loads(request.form['content'])
      lat_lng = latlng_dict['lng'] + "," + latlng_dict['lat']

      try:
          
        j_req = {
        'hit_id': hitId,
        'work_id': workerId,
        'assignmentId': assignmentId, 
        'rvs_sample_number': str(sample_session[workerId]),
        'rvs_path': rvs_path,
        'predict_goal_point': lat_lng,
        'rvs_start_point': sample['rvs_start_point'],
        'task': task_session[session_id],
        'date_start': str(start_session[session_id]),
        'date_finish': str(datetime.utcnow())
        }

        # save to database
        id = workerId + hitId + str(task_session[session_id])

        if 'sandbox' in turkSubmitTo:
          verification_ref_sandbox.document(id).set(j_req)
        else:
          verification_ref.document(id).set(j_req)
      except Exception as e:
        return f"An Error Occured: {e}"

      
      sample_session[workerId] = random.randint(0, size_dataset-1)
      
      task_session[session_id] = task_session[session_id] + 1

      if task_session[session_id]>N_TASKS_PER_USER:
        task_session[session_id] = 1
        address = turkSubmitTo + '/mturk/externalSubmit'
        fullUrl = address + '?assignmentId=' + assignmentId + '&workerId=' + workerId + "&hitId=" + hitId
        return render_template(
          'end.html', 
          bar=100, 
          fullUrl=fullUrl,
        )
      else:
        return home()
      


  start_session[session_id] = datetime.utcnow()

  if task_session[session_id] == 0:
    task_bar = 0
  else:
    task_bar = task_session[session_id] - 1
  progress_task = round(task_bar / N_TASKS_PER_USER * 100)
  
  title = 0 if task_session[session_id]==1 else task_session[session_id]

  return render_template('follower_task.html',
                        start_point=start_point,
                        nav_instruction=nav_instruction,
                        bar=progress_task,
                        title=title,
                        turkSubmitTo=turkSubmitTo,
                        n_sample=n_sample,
                        hitId=hitId,
                        workerId=workerId,
                        assignmentId=assignmentId,
                        session_id=session_id
                        )


@app.errorhandler(500)
def internal_server_error(e):
  return jsonify(error=str(e)), 500

@app.errorhandler(404)
def page_not_found(e):
  return render_template("404.html", exc = e)

@app.route('/map/<n_sample>', methods=['GET', 'POST'])
@app.route('/map')
def map():
  n_sample = request.args.get("n_sample") 
  try:
    return render_template(f'map_{n_sample}.html')
  except:
    # sample = random.randint(0, size_dataset)
    # folium_map, instruction, landmarks, entity = osm_maps_instructions[sample]
    # folium_map.save(path_map)
    return render_template(f'map_{n_sample}.html')


port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':

  app.run(threaded=True, host='0.0.0.0', port=port, debug=True)