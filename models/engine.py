import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class RecommendationEngine:

    def __init__(self):
        self.fashion = None
        self.events = None
        self.df = None
        self.X = None
        self.pid = None
        self.cats = None
        self.pop_norm = None
        self.tfidf_matrix = None
        self.encoders = {}
        self.scaler = None
        self.is_ready = False

    def load(self):
        self._load_data()
        self._preprocess()
        self._build_feature_matrix()
        self._build_tfidf()
        self._build_popularity()
        self.is_ready = True

    def _load_data(self):
        fashion_path = os.getenv("FASHION_CSV", "data/amazon_fashion_50k_graduation.csv")
        events_path = os.getenv("EVENTS_CSV", "data/events.csv")
        self.fashion = pd.read_csv(fashion_path).head(5000)
        self.events = pd.read_csv(events_path)
        print(f"   Fashion : {len(self.fashion):,} products")
        print(f"   Events  : {len(self.events):,} rows")

    def _preprocess(self):
        df = self.fashion.copy()
        if "review_count" in df.columns:
            df["review_count"] = df["review_count"].fillna(0)
        for col in ["category", "brand", "color", "size", "material"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        self.scaler = MinMaxScaler()
        df[["price_norm", "rating_norm"]] = self.scaler.fit_transform(df[["price", "rating"]])
        self.df = df

    def _build_feature_matrix(self):
        FEATURES = ["category_enc", "brand_enc", "color_enc", "size_enc", "material_enc", "price_norm", "rating_norm"]
        WEIGHTS = [5, 2, 1, 0.5, 1, 2, 2]
        available = [f for f in FEATURES if f in self.df.columns]
        weights = [WEIGHTS[FEATURES.index(f)] for f in available]
        self.X = self.df[available].values * weights
        self.pid = self.df["product_id"].values
        self.cats = self.df["category"].values
        print(f"   Feature matrix: {self.X.shape}")

    def _build_tfidf(self):
        self.df["soup"] = self.df["product_name"].astype(str) + " " + self.df["category"].astype(str)
        tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tfidf.fit_transform(self.df["soup"].fillna(""))
        print(f"   TF-IDF matrix : {self.tfidf_matrix.shape}")

    def _build_popularity(self):
        ev = self.events.copy()
        if ev.empty or "itemid" not in ev.columns:
            self.df["pop_norm"] = 0.0
            self.pop_norm = self.df["pop_norm"].values
            print("   Popularity: no events data, defaulting to 0")
            return
        ev["weight"] = ev["event"].map({"view": 1, "addtocart": 3, "transaction": 5})
        pop = ev.groupby("itemid")["weight"].sum().reset_index()
        pop.columns = ["product_id", "popularity_score"]
        self.df = self.df.merge(pop, on="product_id", how="left")
        self.df["popularity_score"] = self.df["popularity_score"].fillna(0)
        self.df["pop_norm"] = self.df["popularity_score"] / (self.df["popularity_score"].max() + 1e-9)
        self.pop_norm = self.df["pop_norm"].values
        print("   Popularity scores computed")

    def recommend_by_preferences(self, categories, min_price=0, max_price=9999, min_rating=0, brands=None, top_n=12):
        mask = (
            self.df["category"].isin(categories)
            & (self.df["price"] >= min_price)
            & (self.df["price"] <= max_price)
            & (self.df["rating"] >= min_rating)
        )
        if brands:
            mask &= self.df["brand"].isin(brands)
        filtered = self.df[mask].copy()
        if filtered.empty:
            return []
        max_reviews = filtered["review_count"].max() if "review_count" in filtered.columns and filtered["review_count"].max() > 0 else 1
        rev_col = filtered["review_count"] / max_reviews if "review_count" in filtered.columns else 0
        filtered["score"] = (
            filtered["rating_norm"] * 0.4
            + rev_col * 0.3
            + (1 - abs(filtered["price_norm"] - filtered["price_norm"].mean())) * 0.3
        )
        cols = [c for c in ["product_id", "product_name", "brand", "category", "price", "rating", "review_count", "color", "size", "material", "score"] if c in filtered.columns]
        result = filtered.sort_values("score", ascending=False).drop_duplicates(subset=["product_name"]).head(top_n)[cols].reset_index(drop=True)
        return result.to_dict(orient="records")

    def find_similar(self, product_id, top_n=6):
        if product_id not in self.pid:
            return None, None
        idx = int(np.where(self.pid == product_id)[0][0])
        product = self.df.iloc[idx]
        same_cat = self.df[self.df["category"] == product["category"]].copy()
        scores = cosine_similarity([self.X[idx]], self.X[same_cat.index])[0]
        same_cat["similarity"] = scores
        cols = [c for c in ["product_id", "product_name", "brand", "category", "price", "rating", "color", "material", "similarity"] if c in same_cat.columns]
        result = same_cat.sort_values("similarity", ascending=False).iloc[1:top_n+1][cols].reset_index(drop=True)
        original = {
            "product_id": int(product["product_id"]),
            "product_name": product["product_name"],
            "category": product["category"],
            "brand": product["brand"],
            "price": float(product["price"]),
            "rating": float(product["rating"]),
        }
        return original, result.to_dict(orient="records")

    def behavioral_recommend(self, interactions, top_n=10):
        EVENT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}
        if not interactions:
            result = self.df.nlargest(top_n, "rating_norm")[["product_id", "product_name", "brand", "category", "price", "rating"]].reset_index(drop=True)
            return result.to_dict(orient="records")
        user_vec = np.zeros(self.X.shape[1])
        weights_sum = 0
        seen_ids = set()
        for item in interactions:
            pid_val = item["product_id"]
            weight = EVENT_WEIGHTS.get(item.get("event", "view"), 1)
            seen_ids.add(pid_val)
            mask = self.pid == pid_val
            if mask.sum() > 0:
                idx = int(np.where(mask)[0][0])
                user_vec += self.X[idx] * weight
                weights_sum += weight
        if weights_sum == 0:
            return []
        user_vec /= weights_sum
        scores = cosine_similarity([user_vec], self.X)[0]
        for pid_val in seen_ids:
            mask = self.pid == pid_val
            if mask.sum() > 0:
                scores[int(np.where(mask)[0][0])] = -1
        top_idx = scores.argsort()[::-1][:top_n]
        result = self.df.iloc[top_idx][["product_id", "product_name", "brand", "category", "price", "rating"]].copy()
        result["match_score"] = scores[top_idx].round(4)
        return result.reset_index(drop=True).to_dict(orient="records")

    def get_kpis(self):
        ev = self.events
        if ev.empty or "event" not in ev.columns:
            return {"total_views": 0, "total_cart_adds": 0, "total_purchases": 0,
                    "conversion_rate": 0, "cart_rate": 0, "abandonment_rate": 0, "abandoned_items": 0}
        counts = ev["event"].value_counts()
        views = int(counts.get("view", 0))
        carts = int(counts.get("addtocart", 0))
        buys = int(counts.get("transaction", 0))
        return {
            "total_views": views,
            "total_cart_adds": carts,
            "total_purchases": buys,
            "conversion_rate": round(buys / views * 100, 2) if views else 0,
            "cart_rate": round(carts / views * 100, 2) if views else 0,
            "abandonment_rate": 0,
            "abandoned_items": 0,
        }

    def get_product_stats(self):
        f = self.fashion
        return {
            "total_products": len(f),
            "total_categories": int(f["category"].nunique()),
            "total_brands": int(f["brand"].nunique()),
            "price_range": {"min": float(f["price"].min()), "max": float(f["price"].max())},
            "avg_price": round(float(f["price"].mean()), 2),
            "avg_rating": round(float(f["rating"].mean()), 2),
            "categories": f["category"].value_counts().to_dict(),
            "avg_price_by_cat": f.groupby("category")["price"].mean().round(2).to_dict(),
            "top_brands": f["brand"].value_counts().head(10).to_dict(),
        }

    def get_categories(self):
        return sorted(self.fashion["category"].unique().tolist())

    def get_brands(self):
        return sorted(self.fashion["brand"].unique().tolist())