from flask import Flask 

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 
app.config['UPLOAD_EXTENSIONS'] = ['.jpeg', '.jpg', '.png', '.gif']

from server import views 