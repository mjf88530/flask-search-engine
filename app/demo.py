from flask import Flask
from flask import render_template
from flask import request
from searcher import run

app = Flask(__name__)


@app.route("/")
def search():
    return render_template('new.html', val='')


@app.route("/search")
def processing():
    user_input = request.args.get('wd')
    result = run(user_input)
    return render_template('new.html', val=user_input, content=result)


if __name__ == '__main__':
    app.run(debug=True)
