from flask import Blueprint, request, jsonify, current_app
import math

behavioral_bp = Blueprint("behavioral", __name__)

def sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj

@behavioral_bp.route("/behavioral/recommend", methods=["POST"])
def behavioral_recommend():
    engine = current_app.config["ENGINE"]
    body   = request.get_json(silent=True) or {}
    interactions = body.get("interactions", [])
    top_n        = int(body.get("top_n", 10))
    valid = []
    for item in interactions:
        if "product_id" not in item:
            continue
        valid.append({
            "product_id": str(item["product_id"]),
            "event": item.get("event", "view")
        })
    results = engine.behavioral_recommend(valid, top_n)
    return jsonify(sanitize({
        "recommendations":   results,
        "count":             len(results),
        "interactions_used": len(valid),
        "mode": "cold_start" if not valid else "personalized",
    }))

@behavioral_bp.route("/behavioral/track", methods=["POST"])
def track():
    body = request.get_json(silent=True) or {}
    required = ["user_id", "product_id", "event"]
    if not all(k in body for k in required):
        return jsonify({"error": f"Required fields: {required}"}), 400
    valid_events = ["view", "addtocart", "transaction"]
    if body["event"] not in valid_events:
        return jsonify({"error": f"event must be one of {valid_events}"}), 400
    return jsonify(sanitize({
        "status": "tracked",
        "user_id": body["user_id"],
        "product_id": str(body["product_id"]),
        "event": body["event"]
    }))