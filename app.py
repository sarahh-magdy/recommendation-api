from flask import Flask
from flask_cors import CORS
from routes.recommend import recommend_bp
from routes.similar import similar_bp
from routes.behavioral import behavioral_bp
from routes.insights import insights_bp
from models.engine import RecommendationEngine

app = Flask(__name__)
CORS(app)

print("Loading BrandHive recommendation engine...")
engine = RecommendationEngine()
engine.load()
print("Engine ready!")

app.config["ENGINE"] = engine

app.register_blueprint(recommend_bp,  url_prefix="/api")
app.register_blueprint(similar_bp,    url_prefix="/api")
app.register_blueprint(behavioral_bp, url_prefix="/api")
app.register_blueprint(insights_bp,   url_prefix="/api")


@app.route("/")
def health():
    return {
        "status": "ok",
        "message": "BrandHive Recommendation API is running!",
        "endpoints": {
            "POST /api/recommend":                "Category-based recommendations",
            "GET  /api/trending":                 "Trending products (optional ?category=)",
            "GET  /api/similar/<product_id>":     "Similar products by MongoDB ID",
            "POST /api/behavioral/recommend":     "Personalized recommendations from interactions",
            "POST /api/behavioral/track":         "Track user event",
            "GET  /api/categories":               "All categories",
            "GET  /api/brands":                   "All brands (optional ?category=)",
            "GET  /api/insights/kpis":            "Platform KPIs",
            "GET  /api/insights/products":        "Product statistics",
            "GET  /api/insights/trending":        "Trending insights",
        }
    }


import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)