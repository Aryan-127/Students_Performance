from flask import Flask, request, render_template
import pickle
import os
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
model_path = os.path.join(BASE_DIR, 'model.pkl')

scaler = pickle.load(open(scaler_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

@app.route('/', methods=["GET", "POST"])
def homepage():
    if request.method == "GET":
        return render_template(
            "index.html",
            form_data={},
            prediction=None,
            error=None
        )

    form_data = request.form.to_dict()
    
    lst = [form_data[val] for val in form_data]

 
    transformed_data = scaler.transform([lst])
    results = model.predict(transformed_data)
    
    return render_template(
        "index.html",
        form_data=form_data,
        prediction=results[0],
        error=None
    )

if __name__ == "__main__":
    app.run(debug=True)
