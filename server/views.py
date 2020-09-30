from server import app

import json 

@app.route('/')
def heartbeat():
    return json.dumps(
        {
            'status': 'SUCCESS', 
            'contents': 'The server is UP and ready to respond'
    })