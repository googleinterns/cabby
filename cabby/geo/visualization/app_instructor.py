from datetime import datetime
from datetime import timedelta
import flask
from flask import Flask
from flask_session import Session
import firebase_admin
from firebase_admin import firestore

import os
import pandas as pd
import random
from shapely.geometry.point import Point
import uuid


from forms import NavigationForm, ReviewForm
import util
import visualize

# Initialize Firestore DB
cred = firebase_admin.credentials.Certificate('key.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()
instructions_ref = db.collection('instructions')
instructions_ref_sandbox = db.collection('instructions_sandbox')

verification_ref = db.collection('verification')
verification_ref_sandbox = db.collection('verification_sandbox')

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'

app.config['SECRET_KEY'] = uuid.uuid4().hex

app.app_context().push()

Session(app)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)


try:
  rvs_path = os.path.abspath("./data/manhattan_samples_v24.gpkg")
except Exception as e:
  print (f"An Error Occured: {e}, {rvs_path}")

# Raw Data
osm_maps_instructions = visualize.get_maps_and_instructions(rvs_path, with_path=False)

size_dataset = len(osm_maps_instructions)

task_session = {}
sample_session = {}
start_session = {}


N_TASKS_PER_USER = 2

dir_map = os.path.join(app.root_path,"templates")

@app.route("/", methods=['GET', 'POST'])
@app.route("/pub/", methods=['GET', 'POST'])
@app.route("/home/", methods=['GET', 'POST'])
def home():
  global sample
  global task_session

  assignmentId = flask.request.args.get("assignmentId") 
  assignmentId = assignmentId if assignmentId else uuid.uuid4().hex
  hitId = flask.request.args.get("hitId")
  hitId = hitId if hitId else "2"
  turkSubmitTo = flask.request.args.get("turkSubmitTo")
  workerId = flask.request.args.get("workerId")
  workerId = workerId if workerId else "1"
  turkSubmitTo = flask.request.args.get("turkSubmitTo")
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
  folium_map, _, landmarks, entity = osm_maps_instructions[
    sample_session[workerId]]
  path_map = os.path.join(dir_map,f"map_{sample_session[workerId]}.html") 
  if os.path.exists(path_map):
    os.remove(path_map)

  folium_map.save(path_map)


  form_nav = NavigationForm()

  form_nav.landmarks = landmarks


  if flask.request.method == 'POST': 
    if os.path.exists(path_map):
      os.remove(path_map)
    folium_map.save(path_map)
    
    if form_nav.validate_on_submit() or (
      len(form_nav.errors)==1 and 'csrf_token' in form_nav.errors):

      content = flask.request.form['content']
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
        'key': id,
        'verified_n': 0
        }

        # save to database
        if 'sandbox' in turkSubmitTo:
          instructions_ref_sandbox.document(id).set(j_req)
        else:
          instructions_ref.document(id).set(j_req)
      except Exception as e:
        return f"An Error Occured: {e}"

      
      sample_session[workerId] = random.randint(0, size_dataset-1)

      folium_map, _, landmarks, entity = osm_maps_instructions[sample_session[workerId]]
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
        return flask.render_template(
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

  return flask.render_template('instructor_task.html',
                         form=form_nav,
                         bar=progress_task,
                         title=title,
                         n_sample = sample_session[workerId],
                         workerId=workerId
                         )
                         

def verification_task(
  workerId, turkSubmitTo, session_id, hitId, assignmentId):

  # read from database
  if 'sandbox' in turkSubmitTo:
    instruction_table = instructions_ref_sandbox
  else:
    instruction_table = instructions_ref

  instruction_data_all = list(instruction_table.get())

  instruction_data = [
    e.to_dict() for e in instruction_data_all if 'review' in e.to_dict() and e.to_dict()[
      'review']=='RVS_excellent' and e.to_dict()[
      'work_id']!=workerId]

  instruction_data_df = pd.DataFrame(instruction_data)
  instruction_data_df.sort_values('verified_n', ascending=True, inplace=True)
  
  sample = instruction_data_df.iloc[0]

  sample_session[workerId] = sample['rvs_sample_number']

  path_data = os.path.abspath(
    sample['rvs_path'].replace("/app_instructor", "."))

  entities = util.load_entities(path_data)

  entity = entities[int(sample['rvs_sample_number'])]

  start_point = util.list_yx_from_point(
    entity.geo_landmarks['start_point'].geometry)
  end_point = util.list_yx_from_point(
    entity.geo_landmarks['end_point'].geometry)
  nav_instruction = sample['content']

  if task_session[session_id] == 0:
    task_bar = 0
  else:
    task_bar = task_session[session_id] - 1

  progress_task = round(task_bar / N_TASKS_PER_USER * 100)

  title = 0 if task_session[session_id]==1 else task_session[session_id]
  

  form = ReviewForm()

  if flask.request.method == 'POST': 

    if flask.request.form.get("submit_button"):
      
      latlng_dict = flask.json.loads(
        flask.request.form['latlng'])
      lng = latlng_dict['lng']
      lat = latlng_dict['lat']
      lat_lng = lng + "," + lat

      try:
          
        id = workerId + hitId + str(task_session[session_id])

        point_pred = Point(float(lng), float(lat))

        point_true = util.point_from_str_point(sample['rvs_goal_point'])

        dist = round(util.get_distance_between_points(point_true, point_pred))

        j_req = {
        'hit_id': hitId,
        'work_id': workerId,
        'assignmentId': assignmentId, 
        'rvs_sample_number': str(sample['rvs_sample_number']),
        'rvs_path': sample['rvs_path'],
        'predict_goal_point': lat_lng,
        'rvs_start_point': sample['rvs_start_point'],
        'task': task_session[session_id],
        'date_start': str(start_session[session_id]),
        'date_finish': str(datetime.utcnow()),
        'rvs_goal_point': sample['rvs_goal_point'],
        'key_instruction': sample['key'],
        'key': id,
        'dist_m': dist
        }

        # save to database
        if 'sandbox' in turkSubmitTo:
          verification_ref_sandbox.document(id).set(j_req)
        else:
          verification_ref.document(id).set(j_req)
      except Exception as e:
        return f"An Error Occured: {e}"

      # Update instruction with number of verifications
      id = sample['key'] 
      instruction_table.document(id).update(
        {'verified_n': int(sample['verified_n']+1)})

      sample_session[workerId] = random.randint(0, size_dataset-1)
      
      task_session[session_id] = task_session[session_id] + 1

      if task_session[session_id]>N_TASKS_PER_USER:
        task_session[session_id] = 1
        address = turkSubmitTo + '/mturk/externalSubmit'
        fullUrl = address + '?assignmentId=' + assignmentId + '&workerId=' + workerId + "&hitId=" + hitId
        return flask.render_template(
          'end.html', 
          bar=100, 
          fullUrl=fullUrl,
        )
      else:
        return home()


  path_verf = sample['rvs_path'].replace("/app_instructor", ".")
  osm_maps_verification = visualize.get_maps_and_instructions(
    path_verf, with_path=False)

  _, _, _, entity = osm_maps_verification[
    int(sample_session[workerId])]

  
  start_session[session_id] = datetime.utcnow()

  if task_session[session_id] == 0:
    task_bar = 0
  else:
    task_bar = task_session[session_id] - 1
  progress_task = round(task_bar / N_TASKS_PER_USER * 100)
  
  title = 0 if task_session[session_id]==1 else task_session[session_id]

  landmark_main = {}
  landmark_around = {}
  landmark_rest = {} 

  for landmark_type, landmark in entity.geo_landmarks.items():
    if landmark_type=='end_point' or landmark.geometry is None:
      continue

    landmark_geom, desc = visualize.get_landmark_desc_geo(
      landmark=landmark
      )
    if 'main' in landmark_type:
      landmark_main[landmark_type] = (desc, landmark_geom)
    elif 'around' in landmark_type:
      landmark_around[landmark_type] = (desc, landmark_geom)
    else:
      landmark_rest[landmark_type] = (desc, landmark_geom)

  return flask.render_template('follower_task.html',
                        end_point=end_point,
                        start_point=start_point,
                        nav_instruction=nav_instruction,
                        bar=progress_task,
                        title=title,
                        turkSubmitTo=turkSubmitTo,
                        n_sample=sample['rvs_sample_number'],
                        hitId=hitId,
                        workerId=workerId,
                        assignmentId=assignmentId,
                        session_id=session_id,
                        form=form,
                        landmark_main=landmark_main,
                        landmark_around=landmark_around,
                        landmark_rest=landmark_rest
                        )


@app.errorhandler(500)
def internal_server_error(e):
  return flask.jsonify(error=str(e)), 500

@app.errorhandler(404)
def page_not_found(e):
  return flask.render_template("404.html", exc = e)

@app.route('/map/<n_sample>', methods=['GET', 'POST'])
@app.route('/map')
def map():
  n_sample = flask.request.args.get("n_sample") 
  workerId = flask.request.args.get("workerId") 

  try:
    return flask.render_template(f'map_{n_sample}.html')
  except:
    n_sample = random.randint(0, size_dataset-1)
    sample_session[workerId] = n_sample
    folium_map, _, _, _ = osm_maps_instructions[n_sample]
    path_map = os.path.join(dir_map,f"map_{n_sample}.html") 
    folium_map.save(path_map)
    return flask.render_template(f'map_{n_sample}.html')


port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':

  app.run(threaded=True, host='0.0.0.0', port=port, debug=True)