from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length


class NavigationForm(FlaskForm):
    content = TextAreaField('Content', validators=[DataRequired(), Length(min=80)])
    submit = SubmitField('Submit and go to next task')