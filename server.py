from flask import Flask, url_for, render_template, jsonify

app = Flask(__name__)
app_config = {
    'port': 8081,
}


@app.route('/')
def main():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(**app_config)
