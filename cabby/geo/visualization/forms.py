from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length


class NavigationForm(FlaskForm):
    content = TextAreaField('Content', validators=[DataRequired(), Length(min=50)])
    submit = SubmitField('Submit Navigation Description')