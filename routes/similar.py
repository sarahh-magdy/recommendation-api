from flask import Blueprint, request, jsonify, current_app
import math

similar_bp = Blueprint("similar", __name__)

def sanitize(obj):
    """Replace NaN/Inf with None recursively"""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj

@similar_bp.route("/similar/<string:product_id>", methods=["GET"])
def similar(product_id):
    engine = current_app.config["ENGINE"]
    top_n  = int(request.args.get("top_n", 6))
    original, similar_products = engine.find_similar(product_id, top_n)
    if original is None:
        return jsonify({"error": f"Product {product_id} not found"}), 404
    return jsonify(sanitize({
        "original": original,
        "similar":  similar_products,
        "count":    len(similar_products)
    }))