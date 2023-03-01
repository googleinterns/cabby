from datetime import datetime
from datetime import timedelta
from operator import iconcat
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
import sys
import time
import traceback
import json


from forms import NavigationForm, ReviewForm
import util
import visualize
import tasks


REGION = 'Manhattan'
DIST_THRESHOLD = 100
N_TASKS_PER_USER = 3
N_VALIDATION = 2
DEFAULT_ARGS = "1A"
SANDBOX_ON = False



def log_info(content):

  j_req = {
  'content': content,
  'date': str(datetime.utcnow()),
  }
  # save to database
  logs.document(str(random.randint(0,100000))).set(j_req)

try:
  rvs_path = os.path.abspath("./data/manhattan_samples_v67.gpkg")
except Exception as e:
  print (f"An Error Occured: {e}")

# Initialize Firestore DB
cred = firebase_admin.credentials.Certificate('key.json')
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()
instructions_ref = db.collection('instructions')
instructions_ref_sandbox = db.collection('instructions_sandbox')


instructions_ref_get = list(instructions_ref.get()) 
instructions_ref_sandbox_get = list(instructions_ref_sandbox.get()) 

verification_ref = db.collection('verification')
verification_ref_sandbox = db.collection('verification_sandbox')


logs = db.collection('logs')

global_variables_db = db.collection('global_variables')


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'

app.config['SECRET_KEY'] = uuid.uuid4().hex

app.app_context().push()

Session(app)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

qualified_workers = pd.read_csv(
  "qualified_workers.csv", header=None)[0].tolist()

# Raw Data
osm_maps_instructions = visualize.get_maps_and_instructions(rvs_path, with_path=False)

size_dataset = len(osm_maps_instructions)

def update_global_variables(global_variables, session_id):
  dict = global_variables.to_dict()
  global_variables_db.document(session_id).set(dict)

dir_map = os.path.join(app.root_path,"templates")

@app.route("/", methods=['GET', 'POST'])
@app.route("/home/", methods=['GET', 'POST'])
def home():
  
  assignmentId = flask.request.args.get("assignmentId") 
  assignmentId = assignmentId if assignmentId else uuid.uuid4().hex
  hitId = flask.request.args.get("hitId")
  hitId = hitId if hitId else str(datetime.now().hour)
  turkSubmitTo = flask.request.args.get("turkSubmitTo")
  workerId = flask.request.args.get("workerId")
  workerId = workerId if workerId else DEFAULT_ARGS
  turkSubmitTo = turkSubmitTo if turkSubmitTo else "https://workersandbox.mturk.com"
  
  session_id = str(workerId+hitId)

  try:     

    global_variables = tasks.Tasks(sessions_id=session_id, size_dataset=size_dataset, n_validation=N_VALIDATION)    

    global_variables = init_global_variable(global_variables, session_id)    
   
    if global_variables.finished == True: 
      return end_hit(global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId)

    if global_variables.task_n>N_TASKS_PER_USER:

      return end_hit(global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId)

    if global_variables.task_n <=N_VALIDATION:

      return verification_task(
        global_variables=global_variables,
        workerId=workerId, 
        hitId=hitId, 
        assignmentId=assignmentId, 
        session_id=session_id, 
        turkSubmitTo=turkSubmitTo)
    
    return description_task(
        global_variables=global_variables,
        workerId=workerId, 
        hitId=hitId, 
        assignmentId=assignmentId, 
        session_id=session_id, 
        turkSubmitTo=turkSubmitTo)
    
  except Exception as e:
    return handle_error(
      global_variables, e, workerId, hitId, session_id, assignmentId, turkSubmitTo)

def init_global_variable(global_variables, session_id):
  if not global_variables.validations[global_variables.task_n-1].rvs_path:
    global_variables_specific = global_variables_db.document(session_id)
    doc = global_variables_specific.get()

    if doc.to_dict():
      global_variables = tasks.Tasks.from_dict(
        doc=doc.to_dict(), size_dataset=size_dataset, n_validation=N_VALIDATION)
    else:
      global_variables = tasks.Tasks(
        sessions_id=session_id, size_dataset=size_dataset, n_validation=N_VALIDATION)    
      global_variables_specific.set(global_variables.to_dict())

  return global_variables


def end_hit(
  global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId):

  address = turkSubmitTo + '/mturk/externalSubmit'
  fullUrl = address + '?assignmentId=' + assignmentId #+ '&workerId=' + workerId + "&hitId=" + hitId

  try:
    global_variables.finished = True
  except:
    write_error_to_log(global_variables, workerId, hitId, session_id, assignmentId)

  return flask.render_template(
    'end.html', 
    bar=100, 
    fullUrl=fullUrl,
  )


def write_log(global_variables, workerId, hitId, session_id, assignmentId, content):

  id = str(random.randint(0,1000000))

  j_req = {
    'date': str(datetime.utcnow()),
    'hit_id': hitId,
    'work_id': workerId,
    'session_id': session_id,
    'assignmentId': assignmentId, 
    'key': id,
    'content': content,
    'url': flask.request.url,
    }

  validation_log = ""
  if global_variables and global_variables.session_id:   
    if  global_variables.task_n-1<len(global_variables.validations):
      validation_log = global_variables.validations[global_variables.task_n-1].rvs_path
    j_req['task_session_n'] = global_variables.task_n
    j_req['sample_session_n'] = global_variables.write.writing_task_n
    j_req['start_session'] = global_variables.start_session

  j_req['validation_session'] = validation_log
  # save to database
  logs.document(id).set(j_req)


def write_error_to_log(global_variables, workerId, hitId, session_id, assignmentId):
  # Get current system exception

  _, _, ex_traceback = sys.exc_info()

  # Extract unformatter stack traces as tuples
  trace_back = traceback.extract_tb(ex_traceback)

  # Format stacktrace
  log_state = 'Error LOG'
  if 'ASSIGNMENT_ID_NOT_AVAILABLE' in flask.request.url:
    log_state = 'Warning LOG'
  stack_trace = [log_state]

  for trace in trace_back:
      stack_trace.append(
        "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
  
  write_log(global_variables, workerId, hitId, session_id, assignmentId, stack_trace)
  print (stack_trace)

def handle_error(
  global_variables, error, workerId, hitId, session_id, assignmentId, turkSubmitTo):

  write_error_to_log(global_variables, workerId, hitId, session_id, assignmentId)

  if 'ASSIGNMENT_ID_NOT_AVAILABLE' in flask.request.url:
    return home()
  return end_hit(global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId)

def get_entity_verification(
  global_variables, instruction_data_df, workerId, hitId, session_id, assignmentId):

  instruction_data_df.sort_values('verified_n', ascending=True, inplace=True)
  not_validated = instruction_data_df[instruction_data_df['valid']==False]

  if not_validated.shape[0]!=0:
    min_verified_n = not_validated.iloc[0]['verified_n']
    not_validated_min = not_validated[not_validated['verified_n']==min_verified_n]

    sample = not_validated_min.sample(1).iloc[0]
  else:
    instruction_data_df.sort_values('verified_n', ascending=True, inplace=True)
    sample = instruction_data_df.iloc[0]
  

  path_data = os.path.abspath(
    sample['rvs_path'].replace("/app_instructor", "."))

    
  entities = util.load_entities(path_data)

  sample_n = int(sample['rvs_sample_number'])

  try:
    entity = entities[sample_n]
  except Exception as e:
    tb = e.__traceback__

    id = workerId + hitId + str(global_variables.task_n)
    j_req = {
        'date': str(datetime.utcnow()),
        'hit_id': hitId,
        'work_id': workerId,
        'session_id': session_id,
        'assignmentId': assignmentId, 
        'key': id,
        'content': f"Verification error line: {tb.tb_lineno}",
        }

    # save to database
    logs.document(id).set(j_req)
        
    instruction_data_df.sort_values('verified_n', ascending=True, inplace=True)
    sample = instruction_data_df.iloc[0]

    path_data = os.path.abspath(sample['rvs_path'].replace("/app_instructor", "."))

    entities = util.load_entities(path_data)
    entity = entities[sample_n]

  return entity, path_data, sample_n, sample

def update_task_number(global_variables, session_id):
  global_variables.task_n += 1
  update_global_variables(global_variables, session_id)

def post_descriptions(
  global_variables,
  entity, 
  folium_map, 
  path_map,
  workerId, 
  hitId, 
  session_id, 
  form_nav, 
  assignmentId, 
  turkSubmitTo):

  if os.path.exists(path_map):
    os.remove(path_map)
  folium_map.save(path_map)
  
  if form_nav.validate_on_submit() or (
    len(form_nav.errors)==1 and 'csrf_token' in form_nav.errors):

    content = flask.request.form['content']
    goal_point = entity.geo_landmarks['end_point'].geometry
    start_point = entity.geo_landmarks['start_point'].geometry

    try:
      id = workerId + hitId + str(global_variables.task_n)

      j_req = {
      'hit_id': hitId,
      'work_id': workerId,
      'assignmentId': assignmentId, 
      'rvs_sample_number': str(global_variables.write.writing_task_n),
      'content': content,
      'rvs_path': rvs_path,
      'rvs_goal_point': [goal_point.y, goal_point.x],
      'rvs_start_point': [start_point.y, start_point.x],
      'task': global_variables.task_n,
      'date_start': str(global_variables.start_session),
      'date_finish': str(datetime.utcnow()),
      'key': id,
      'valid': False,
      'verified_n': 0,
      'region': REGION
      }


      # save to database
      if 'sandbox' in turkSubmitTo and SANDBOX_ON:
        instructions_ref_sandbox.document(id).set(j_req)
      else:
        instructions_ref.document(id).set(j_req)

      # write_log(workerId, hitId, session_id, assignmentId, f"Writing task - update data: {j_req}" )
      global_variables.finished = True
      update_global_variables(global_variables, session_id)

    except Exception as e:
      return f"An Error Occured: {e}"

    
    global_variables.write.writing_task_n = random.randint(0, size_dataset-1)

    folium_map, _, landmarks, entity, _ = osm_maps_instructions[global_variables.write.writing_task_n]
    path_map = os.path.join(dir_map,f"map_{global_variables.write.writing_task_n}.html") 
    if os.path.exists(path_map):
      os.remove(path_map)
    folium_map.save(path_map)

    form_nav = NavigationForm()
    form_nav.landmarks = landmarks
    
    update_task_number(global_variables, session_id)
    
    if global_variables.task_n>N_TASKS_PER_USER:
      return end_hit(global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId)
    else:
      return home()


def update_validation_session(
  global_variables,
  sample, 
  workerId, 
  start_point, 
  end_point, 
  nav_instruction, 
  icon_path, 
  landmark_rest, 
  landmark_around, 
  landmark_main):

  workerId = flask.request.args.get("workerId") 
  hitId = flask.request.args.get("hitId") 
  workerId = workerId if workerId else DEFAULT_ARGS
  hitId = hitId if hitId else str(datetime.now().hour)
  
  session_id = str(workerId+hitId)

  global_variables = init_global_variable(global_variables, session_id)
  

  n_validation = len(global_variables.validations)
  if n_validation<global_variables.task_n:
    global_variables.task_n = n_validation

  if not global_variables.validations[global_variables.task_n-1].rvs_sample_number:

    global_variables.validations[global_variables.task_n-1].rvs_sample_number = sample['rvs_sample_number']
    global_variables.validations[global_variables.task_n-1].rvs_path = sample['rvs_path']
    global_variables.validations[global_variables.task_n-1].rvs_start_point = start_point
    global_variables.validations[global_variables.task_n-1].rvs_goal_point = end_point
    global_variables.validations[global_variables.task_n-1].nav_instruction = nav_instruction
    global_variables.validations[global_variables.task_n-1].icon_path = icon_path
    global_variables.validations[global_variables.task_n-1].landmark_rest = landmark_rest
    global_variables.validations[global_variables.task_n-1].landmark_around = landmark_around
    global_variables.validations[global_variables.task_n-1].landmark_main = landmark_main
    global_variables.validations[global_variables.task_n-1].key = sample['key'] 
    global_variables.validations[global_variables.task_n-1].verified_n = sample['verified_n'].item()
    
    
  update_global_variables(global_variables, session_id)
  return global_variables

def post_verification(
  global_variables, workerId, hitId, session_id, assignmentId, turkSubmitTo, instruction_table):

  latlng_dict = flask.json.loads(
    flask.request.form['latlng'])
  lng = float(latlng_dict['lng'])
  lat = float(latlng_dict['lat'])
  predicted_point = [lat, lng]

    
  id = workerId + hitId + str(global_variables.task_n) + str(random.randint(0,1000000))

  point_pred = Point(float(lng), float(lat))

  point_true = util.point_from_list_coord_yx(global_variables.validations[global_variables.task_n-1].rvs_goal_point)

  dist = round(util.get_distance_between_points(point_true, point_pred))

  valid = False if dist>DIST_THRESHOLD else True
  j_req = {
  'hit_id': hitId,
  'work_id': workerId,
  'assignmentId': assignmentId, 
  'rvs_sample_number': str(global_variables.validations[global_variables.task_n-1].rvs_sample_number),
  'rvs_path': global_variables.validations[global_variables.task_n-1].rvs_path,
  'predict_goal_point': predicted_point,
  'rvs_start_point': global_variables.validations[global_variables.task_n-1].rvs_start_point,
  'task': global_variables.task_n,
  'date_start': str(global_variables.start_session),
  'date_finish': str(datetime.utcnow()),
  'rvs_goal_point': global_variables.validations[global_variables.task_n-1].rvs_goal_point,
  'key_instruction': global_variables.validations[global_variables.task_n-1].key,
  'key': id,
  'dist_m': dist,
  }

  # save to database
  if 'sandbox' in turkSubmitTo and SANDBOX_ON:
    verification_ref_sandbox.document(id).set(j_req)
  else:
    verification_ref.document(id).set(j_req)

  # Update instruction with number of verifications
  id = global_variables.validations[global_variables.task_n-1].key
  instruction_table.document(id).update(
    {
      'verified_n': int(global_variables.validations[global_variables.task_n-1].verified_n+1),
      'valid': valid})

  update_task_number(global_variables, session_id)
  print ("Updated Verification task.")
  if global_variables.task_n>N_TASKS_PER_USER:
    global_variables.task_n = 1
    address = turkSubmitTo + '/mturk/externalSubmit'
    fullUrl = address + '?assignmentId=' + assignmentId + '&workerId=' + workerId + "&hitId=" + hitId
    return flask.render_template(
      'end.html', 
      bar=100, 
      fullUrl=fullUrl,
    )

  else:
    return home()


def get_landmarks_verification(entity):

  landmark_main = {}
  landmark_around = {}
  landmark_rest = {} 

  for landmark_type, landmark in entity.geo_landmarks.items():
    if landmark_type=='end_point' or landmark.geometry is None:
      continue

    landmark_geom, desc = visualize.get_landmark_desc_geo(
      landmark=landmark,
      landmark_type=landmark_type,
      )
    if 'main' in landmark_type:
      landmark_main[landmark_type] = {}
      landmark_main[landmark_type]["desc"] = desc
      landmark_main[landmark_type]["point"] = landmark_geom
    elif 'around' in landmark_type:
      landmark_around[landmark_type] = {}
      landmark_around[landmark_type]["desc"] = desc
      landmark_around[landmark_type]["point"] = landmark_geom
    else:
      landmark_rest[landmark_type] = {} 
      landmark_rest[landmark_type]["desc"] = desc
      landmark_rest[landmark_type]["point"] = landmark_geom
    
  return landmark_main, landmark_around, landmark_rest

def update_task_bar_and_title(global_variables, session_id):

    if global_variables.task_n == 0:
      task_bar = 0
    else:
      task_bar = global_variables.task_n - 1
    progress_task = round(task_bar / N_TASKS_PER_USER * 100)
    
    title = 0 if global_variables.task_n==1 else global_variables.task_n

    return progress_task, title


def description_task(
  global_variables, workerId, hitId, assignmentId, session_id, turkSubmitTo):

  try:
    n_sample = global_variables.write.writing_task_n

    folium_map, _, landmarks, entity, _ = osm_maps_instructions[n_sample]

    path_map = os.path.join(dir_map,f"map_{global_variables.write.writing_task_n}.html") 
    if os.path.exists(path_map):
      os.remove(path_map)

    folium_map.save(path_map)


    form_nav = NavigationForm()

    form_nav.landmarks = landmarks

    if flask.request.method == 'POST' and global_variables.write.post: 
      return post_descriptions(
        global_variables,
        entity, 
        folium_map, 
        path_map, 
        workerId, 
        hitId, 
        session_id, 
        form_nav, 
        assignmentId, 
        turkSubmitTo)

    global_variables.start_session = str(datetime.utcnow())

    progress_task, title = update_task_bar_and_title(global_variables, session_id)

    workerId = flask.request.args.get("workerId") 
    workerId = workerId if workerId else DEFAULT_ARGS

    session_id = str(workerId+hitId)
    
    global_variables.write.post = True

    update_global_variables(global_variables, session_id)

    n_sample = global_variables.write.writing_task_n

    return flask.render_template('instructor_task.html',
                          form=form_nav,
                          bar=progress_task,
                          title=title,
                          n_sample = str(n_sample),
                          workerId=workerId
                          )

  except Exception as e:
    return handle_error(
      global_variables, e, workerId, hitId, session_id, assignmentId, turkSubmitTo)                        

def verification_task(
  global_variables, workerId, turkSubmitTo, session_id, hitId, assignmentId):
 
  try: 
    
    if not global_variables or not global_variables.session_id:
      return home()

    # read from database
    if 'sandbox' in turkSubmitTo and SANDBOX_ON:
      instruction_table = instructions_ref_sandbox
      instruction_data_all = instructions_ref_sandbox_get
    else:
      instruction_table = instructions_ref
      instruction_data_all = instructions_ref_get
    
    if global_variables.validations[global_variables.task_n-1].rvs_path and global_variables.validations[global_variables.task_n-1].finished: #and workerId:
      try:
        latlng_dict = flask.json.loads(flask.request.form['latlng'])
      except:
        latlng_dict = False

      if latlng_dict: #flask.request.method == 'POST' and latlng_dict: 

        # if flask.request.form.get("submit_button"):

          global_variables.validations[global_variables.task_n-1].finished = False
          update_global_variables(global_variables, session_id)
          return post_verification(
            global_variables, workerId, hitId, session_id, assignmentId, turkSubmitTo, instruction_table)
      
    else: 
      instruction_data = [
        e.to_dict() for e in instruction_data_all if (
          workerId in qualified_workers or (
            'review' in e.to_dict() and e.to_dict()[
              'review']=='RVS_excellent')) and e.to_dict()[
                'work_id']!=workerId]


      instruction_data_df = pd.DataFrame(instruction_data)
      entity, path_data, sample_n, sample = get_entity_verification(
        global_variables, instruction_data_df, workerId, hitId, session_id, assignmentId)

      start_point = util.list_yx_from_point(
        entity.geo_landmarks['start_point'].geometry)
      end_point = util.list_yx_from_point(
        entity.geo_landmarks['end_point'].geometry)
      nav_instruction = sample['content']
      
      osm_maps_verification = visualize.get_maps_and_instructions(
        path_data, with_path=False, specific_sample=sample_n)

      _, _, _, entity, icon_path = osm_maps_verification[0]

      if icon_path:
        icon_path = icon_path.split('osm_icons/')[-1]

      global_variables.start_session = str(datetime.utcnow())

      progress_task, title = update_task_bar_and_title(global_variables, session_id)

      landmark_main, landmark_around, landmark_rest = get_landmarks_verification(entity)

      global_variables = update_validation_session(
        global_variables,
        sample,
        workerId, 
        start_point, 
        end_point, 
        nav_instruction, 
        icon_path, 
        landmark_rest, 
        landmark_around, 
        landmark_main)
            

    progress_task = 0
    title = 0
    
    if global_variables and session_id:
      progress_task, title = update_task_bar_and_title(global_variables, session_id)
    form = ReviewForm()

    if global_variables.task_n <= len(global_variables.validations):

      global_variables.validations[global_variables.task_n-1].finished = True
      update_global_variables(global_variables, session_id)

    return flask.render_template('follower_task.html',
                          end_point=global_variables.validations[global_variables.task_n-1].rvs_goal_point,
                          start_point=global_variables.validations[global_variables.task_n-1].rvs_start_point,
                          nav_instruction=global_variables.validations[global_variables.task_n-1].nav_instruction,
                          bar=progress_task,
                          title=title,
                          form=form,
                          landmark_main=global_variables.validations[global_variables.task_n-1].landmark_main,
                          landmark_around=global_variables.validations[global_variables.task_n-1].landmark_around,
                          landmark_rest=global_variables.validations[global_variables.task_n-1].landmark_rest,
                          icon_path=global_variables.validations[global_variables.task_n-1].icon_path
                          )

  except Exception as e:
    return handle_error(
      global_variables, e, workerId, hitId, session_id, assignmentId, turkSubmitTo)
  
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

  try:
    return flask.render_template(f'map_{n_sample}.html')
  except:
    return home()

port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
  app.run(threaded=True, host='0.0.0.0', port=port, debug=False)
