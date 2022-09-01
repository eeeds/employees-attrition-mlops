from flask import Flask, render_template

app = Flask(__name__, template_folder='.')


@app.route('/')
def evidently():
    return render_template('df_model_performance.html')


if __name__ == '__main__':
    app.run(debug=True)
