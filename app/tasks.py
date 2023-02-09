import random
from datetime import datetime

class Tasks():
  def __init__(self, size_dataset, sessions_id=None):
    
    self.session_id = sessions_id
    self.start_session = str(datetime.utcnow()) 
    self.task_n = 1

    self.finished = False

    self.validation = Validarion()

    self.write = Write(size_dataset=size_dataset)

  def to_dict(self):
    task_dict = {}
    task_dict['write'] = self.write.__dict__

    task_dict['session_id'] = self.session_id
    task_dict['start_session'] = self.start_session
    task_dict['task_n'] = self.task_n
    task_dict['finished'] = self.finished
    task_dict['validation'] = {}

    for k, v in self.validation.__dict__.items():
        task_dict['validation'][k] = v

    return task_dict

  @staticmethod
  def from_dict(doc, size_dataset):
    task = Tasks(sessions_id=doc['session_id'], size_dataset=size_dataset)
    task.start_session = doc['start_session']
    task.task_n = doc['task_n']
    task.finished = doc['finished']
    task.validation = Validarion(doc['validation'])
    task.write = Write(my_dict=doc['write'], size_dataset=size_dataset)
    return task
    

class Validarion():
  def __init__(self, my_dict=None):
    if my_dict:
      for key in my_dict:
        setattr(self, key, my_dict[key])

    else:
      self.rvs_sample_number = None 
      self.rvs_path = None
      self.rvs_start_point = None
      self.rvs_goal_point = None
      self.nav_instruction= None
      self.icon_path = None
      self.landmark_rest = None
      self.landmark_around = None
      self.landmark_main = None
      self.key = None
      self.verified_n = None
      
      self.finished = False


class Write():
  def __init__(self, size_dataset, my_dict=None):
    if my_dict:
      for key in my_dict:
          setattr(self, key, my_dict[key])
    else: 
      print ("size_dataset ", size_dataset)
      x = random.randint(0, size_dataset-1)
      self.writing_task_n = random.randint(0, size_dataset-1)

      self.finished = False
      self.post = False

