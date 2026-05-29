from flask import Blueprint, request, jsonify, current_app
import math

recommend_bp = Blueprint("recommend", __name__)

def sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj

@recommend_bp.route("/recommend", methods=["POST"])
def recommend():
    engine = current_app.config["ENGINE"]
    body   = request.get_json(silent=True) or {}
    categories = body.get("categories", [])
    if not categories:
        return jsonify({"error": "categories is required"}), 400
    results = engine.recommend_by_preferences(
        categories   = categories,
        min_price    = float(body.get("min_price", 0)),
        max_price    = float(body.get("max_price", 999999)),
        min_rating   = float(body.get("min_rating", 0)),
        brands       = body.get("brands"),
        on_sale_only = bool(body.get("on_sale_only", False)),
        top_n        = int(body.get("top_n", 12)),
    )
    return jsonify(sanitize({"products": results, "count": len(results)}))

@recommend_bp.route("/trending", methods=["GET"])
def trending():
    engine   = current_app.config["ENGINE"]
    category = request.args.get("category")
    top_n    = int(request.args.get("top_n", 12))
    results  = engine.get_trending(category=category, top_n=top_n)
    return jsonify(sanitize({"products": results, "count": len(results)}))

@recommend_bp.route("/categories", methods=["GET"])
def get_categories():
    engine = current_app.config["ENGINE"]
    return jsonify({"categories": engine.get_categories()})

@recommend_bp.route("/brands", methods=["GET"])
def get_brands():
    engine   = current_app.config["ENGINE"]
    category = request.args.get("category")
    brands   = engine.get_brands_by_category(category) if category else engine.get_brands()
    return jsonify({"brands": brands})