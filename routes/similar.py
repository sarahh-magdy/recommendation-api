from flask import Blueprint, request, jsonify, current_app

similar_bp = Blueprint("similar", __name__)


@similar_bp.route("/similar/<int:product_id>", methods=["GET"])
def similar(product_id):
    engine = current_app.config["ENGINE"]
    top_n  = int(request.args.get("top_n", 6))
    original, similar_products = engine.find_similar(product_id, top_n)
    if original is None:
        return jsonify({"error": f"Product {product_id} not found"}), 404
    return jsonify({"original": original, "similar": similar_products, "count": len(similar_products)})