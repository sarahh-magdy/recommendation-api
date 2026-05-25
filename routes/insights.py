from flask import Blueprint, jsonify, current_app

insights_bp = Blueprint("insights", __name__)


@insights_bp.route("/insights/kpis", methods=["GET"])
def kpis():
    engine = current_app.config["ENGINE"]
    return jsonify(engine.get_kpis())


@insights_bp.route("/insights/products", methods=["GET"])
def product_stats():
    engine = current_app.config["ENGINE"]
    return jsonify(engine.get_product_stats())