from flask import Blueprint, request, jsonify
from backend.calibration_manager import CalibrationManager

bp = Blueprint("calibration", __name__, url_prefix="/api/calibration")

manager = CalibrationManager()

@bp.route("/run", methods=["POST"])
def run_calibration():
    cam = request.json["camera"]
    result = manager.calibrate(cam)
    return jsonify(result)