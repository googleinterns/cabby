import random
from datetime import datetime

class Tasks():
  def __init__(self, size_dataset=1, sessions_id=None, n_validation=1):
    
    self.session_id = sessions_id
    self.start_session = str(datetime.utcnow()) 
    self.task_n = 1

    self.finished = False

    self.validations = [Validation()]*n_validation

    self.write = Write(size_dataset=size_dataset)

  def to_dict(self):
    task_dict = {}
    task_dict['write'] = self.write.__dict__

    task_dict['session_id'] = self.session_id
    task_dict['start_session'] = self.start_session
    task_dict['task_n'] = self.task_n
    task_dict['finished'] = self.finished
    task_dict['validations'] = []

    for val_idx, validation in enumerate(self.validations):
      task_dict['validations'].append({})
      for k, v in validation.__dict__.items():
          task_dict['validations'][val_idx][k] = v

    return task_dict

  @staticmethod
  def from_dict(doc, size_dataset, n_validation):
    task = Tasks(
      sessions_id=doc['session_id'], size_dataset=size_dataset, n_validation=n_validation)
    task.start_session = doc['start_session']
    task.task_n = doc['task_n']
    task.finished = doc['finished']
    task.validations = []
    for val_idx, validation in enumerate(doc['validations']):
      task.validations.append(Validation(validation))
    task.write = Write(my_dict=doc['write'], size_dataset=size_dataset)
    return task
    

class Validation():
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
  def __init__(self, size_dataset=1, my_dict=None):
    if my_dict:
      for key in my_dict:
          setattr(self, key, my_dict[key])
    else: 
      self.writing_task_n = random.randint(0, size_dataset-1)

      self.finished = False
      self.post = False