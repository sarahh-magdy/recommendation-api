from flask import Blueprint, request, jsonify, current_app

recommend_bp = Blueprint("recommend", __name__)


@recommend_bp.route("/recommend", methods=["POST"])
def recommend():
    engine = current_app.config["ENGINE"]
    body   = request.get_json(silent=True) or {}
    categories = body.get("categories", [])
    if not categories:
        return jsonify({"error": "categories is required"}), 400
    results = engine.recommend_by_preferences(
        categories = categories,
        min_price  = float(body.get("min_price", 0)),
        max_price  = float(body.get("max_price", 9999)),
        min_rating = float(body.get("min_rating", 0)),
        brands     = body.get("brands"),
        top_n      = int(body.get("top_n", 12)),
    )
    return jsonify({"products": results, "count": len(results)})


@recommend_bp.route("/categories", methods=["GET"])
def get_categories():
    engine = current_app.config["ENGINE"]
    return jsonify({"categories": engine.get_categories()})


@recommend_bp.route("/brands", methods=["GET"])
def get_brands():
    engine = current_app.config["ENGINE"]
    return jsonify({"brands": engine.get_brands()})