from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, ValidationError
import nltk
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

  if len(tags)<1:
    raise ValidationError('Try again to describe the goal.')


def check_landmarks(form, field):
  if all(landmark.lower() not in field.data.lower() for landmark in form.landmarks):
    raise ValidationError('Describe the goal relative to the landmarks.')

class NavigationForm(FlaskForm):
  class Meta:
    csrf = True
  content = TextAreaField('Content', 
  validators=[
    DataRequired(),
    Length(min=100),
    check_verbs,
    check_landmarks
  ]
  )
  submit = SubmitField('Submit and go to next task')

class ReviewForm(FlaskForm):
  class Meta:
    csrf = True
  content = TextAreaField('Content', )
  submit = SubmitField('Submit review and finish assignment')

