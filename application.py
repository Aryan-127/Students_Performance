from flask import Flask,request,render_template
import pickle
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app = application

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

    scaler = pickle.load(open('C:/Users/Aryan/OneDrive/Desktop/Python Udemy/Projects/Students_Performance/scaler.pkl', 'rb'))
    model = pickle.load(open('C:/Users/Aryan/OneDrive/Desktop/Python Udemy/Projects/Students_Performance/model.pkl', 'rb'))

    results = model.predict(scaler.transform([lst]))
    return render_template(
        "index.html",
        form_data=form_data,
        prediction=results[0],
        error=None
    )

if(__name__=="__main__"):
    app.run(debug=True)