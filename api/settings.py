from flask import Blueprint, request, jsonify
from backend.hardware_manager import HardwareManager

bp = Blueprint("settings", __name__, url_prefix="/api/settings")

manager = HardwareManager()

@bp.route("/", methods=["GET"])
def get_settings():
    return jsonify(manager.get_all())

@bp.route("/", methods=["POST"])
def update_settings():
    manager.update(request.json)
    return jsonify({"status": "ok"})