import flask as fk
from model.Model import inference

request = fk.request
jsonify = fk.jsonify


app = fk.Flask(__name__)

@app.route("/")
def testPage():
	return fk.render_template("index.html", name="TF")

@app.route("/predict", methods=['POST'])
def predict():
	data = request.json['data'] #{'x':3}
	res = inference(data)
	return jsonify(labels=str(res[0]), values=str(res[1]))

if __name__ == "__main__":
	app.run(host='0.0.0.0')

