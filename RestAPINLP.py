from flask import (
    Flask,
    render_template,
    request
)

import loadModel 

# Create the application instance
app = Flask(__name__)

# Create a URL route in our application for "/"
@app.route('/getentities', methods=['GET'])
def home():
    text = request.args.get('text')
    return loadModel.extractEntities(text)

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(debug=True)