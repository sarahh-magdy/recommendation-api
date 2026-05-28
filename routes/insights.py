from flask import Blueprint, request, jsonify, current_app

insights_bp = Blueprint("insights", __name__)


@insights_bp.route("/insights/kpis", methods=["GET"])
def kpis():
    engine = current_app.config["ENGINE"]
    return jsonify(engine.get_kpis())


@insights_bp.route("/insights/products", methods=["GET"])
def product_stats():
    engine = current_app.config["ENGINE"]
    return jsonify(engine.get_product_stats())


@insights_bp.route("/insights/trending", methods=["GET"])
def trending_insights():
    engine   = current_app.config["ENGINE"]
    category = request.args.get("category")
    top_n    = int(request.args.get("top_n", 10))
    results  = engine.get_trending(category=category, top_n=top_n)
    return jsonify({"trending": results, "count": len(results)})