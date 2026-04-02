from flask import Flask, render_template
from api import cameras, calibration, settings

app = Flask(__name__)

# Register API blueprints
app.register_blueprint(cameras.bp)
app.register_blueprint(calibration.bp)
app.register_blueprint(settings.bp)

# Pages
@app.route("/")
def camera_page():
    return render_template("camera.html")

@app.route("/calibration")
def calibration_page():
    return render_template("calibration.html")

@app.route("/settings")
def settings_page():
    return render_template("settings.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5800, debug=True)