from flask import Blueprint, request, jsonify, current_app
import math

insights_bp = Blueprint("insights", __name__)

def sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj

@insights_bp.route("/insights/kpis", methods=["GET"])
def kpis():
    engine = current_app.config["ENGINE"]
    return jsonify(sanitize(engine.get_kpis()))

@insights_bp.route("/insights/products", methods=["GET"])
def product_stats():
    engine = current_app.config["ENGINE"]
    return jsonify(sanitize(engine.get_product_stats()))

@insights_bp.route("/insights/trending", methods=["GET"])
def trending_insights():
    engine   = current_app.config["ENGINE"]
    category = request.args.get("category")
    top_n    = int(request.args.get("top_n", 10))
    results  = engine.get_trending(category=category, top_n=top_n)
    return jsonify(sanitize({"trending": results, "count": len(results)}))