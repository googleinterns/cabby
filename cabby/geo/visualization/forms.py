from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, ValidationError
import nltk


def check_second_person(form, field):
  if 'you' not in field.data.lower():
    raise ValidationError('Describe the goal as if you are telling a friend about it.')

def check_no_first_person(form, field):
  if 'We ' in field.data or 'I ' in field.data:
    raise ValidationError('Describe the goal as if you are telling a friend about it.')

def check_verbs(form, field):
  # word tokenizeing and part-of-speech tagger
  tokens = [nltk.word_tokenize(sent) for sent in [field.data]]
  postag = [nltk.pos_tag(sent) for sent in tokens][0]
  tags = [pt[1] for pt in postag if 'VB' in pt[1]]

  if len(tags)<1:
    raise ValidationError('Try again to describe the goal.')


def check_landmarks(form, field):
  if all(landmark not in field.data for landmark in form.landmarks):
    raise ValidationError('Describe the goal relative to the landmarks.')

document = 'The little brown dog barked at the black cat'
tokens = [nltk.word_tokenize(sent) for sent in [document]]
postag = [nltk.pos_tag(sent) for sent in tokens][0]

class NavigationForm(FlaskForm):
  content = TextAreaField('Content', validators=[
    DataRequired(),
    Length(min=80),
    check_second_person,
    check_no_first_person,
    check_verbs,
    check_landmarks
  ])
  submit = SubmitField('Submit and go to next task')

