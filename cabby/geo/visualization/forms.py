from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, ValidationError
import nltk
from fuzzywuzzy import fuzz

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def check_second_person(form, field):
  if 'you' not in field.data.lower():
    raise ValidationError('Describe the goal as if you are telling a friend about it.')

def check_no_first_person(form, field):
  if 'We ' in field.data or 'I' in field.data:
    raise ValidationError('Describe the goal as if you are telling a friend about it.')

def check_verbs(form, field):
  # word tokenizeing and part-of-speech tagger
  tokens = [nltk.word_tokenize(sent) for sent in [field.data]]
  postag = [nltk.pos_tag(sent) for sent in tokens][0]
  tags = [pt[1] for pt in postag if 'VB' in pt[1]]

  validation_mess = 'The description should not be a list of the landmarks.'
  # if len(tags)<1:
  #   raise ValidationError(validation_mess)
  
  landmark_names = ','.join(form.landmarks)

  str_cnt_desc = field.data.count(":")

  if str_cnt_desc>3:
    raise ValidationError(validation_mess)
  
  
  str_cnt_list = field.data.count(",")

  if str_cnt_list>8:
    raise ValidationError(validation_mess)


  score_all_landmarks = fuzz.token_sort_ratio(landmark_names, field.data.lower())
  if score_all_landmarks>=80:
    raise ValidationError(validation_mess)



def check_landmarks(form, field):
  if all(landmark.lower() not in field.data.lower() for landmark in form.landmarks):
    raise ValidationError('Describe the goal relative to the landmarks.')


def check_not_instruction(form, field):
  if "a friend of yours is at the location" in field.data.lower():
    raise ValidationError('Do not copy the instructions.')




def check_relative_language(form, field):
  if "turn right" in field.data.lower():
    raise ValidationError(
      "Instead of using terms such as 'turn right' use relative terms between the landmarks and goal.")
  if "turn left" in field.data.lower():
    raise ValidationError(
      "Instead of using terms such as 'turn left' use relative terms between the landmarks and goal.")


class NavigationForm(FlaskForm):
  class Meta:
    csrf = True
  content = TextAreaField('Content', 
  validators=[
    DataRequired(),
    Length(min=100),
    check_verbs,
    check_landmarks,
    check_not_instruction,
    check_relative_language,
  ]
  )
  submit = SubmitField('Submit')

class ReviewForm(FlaskForm):
  class Meta:
    csrf = False
  content = TextAreaField('Content', 
  validators=[
    DataRequired(),
  ])
  submit = SubmitField('Submit review and finish assignment')

