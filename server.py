from data.senta import predict

from train import train, TRAIN_TEST_SPLIT
from data.statistic import TOTAL_NUM, POSITIVE_NUM, NEGATIVE_NUM

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

# CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有域名跨域

app_config = {
    'port': 8081,
    'debug': True
}


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/statistic')
@cross_origin()
def statistic():
    data = {
        'total_num': TOTAL_NUM,
        'train_test_split': TRAIN_TEST_SPLIT,
        'positive_num': POSITIVE_NUM,
        'negative_num': NEGATIVE_NUM,
    }

    return jsonify(data)


metrics = train()


@app.route('/metrics')
@cross_origin()
def get_metrics():
    return jsonify({
        'accuracy': metrics[0],
        'precision': metrics[1],
        'recall': metrics[2],
        'f1score': metrics[3]
    })


@app.route('/predict')
@cross_origin()
def predict_text():
    text = request.args.get('text')
    result = predict(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(**app_config)
