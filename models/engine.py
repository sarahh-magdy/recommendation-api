import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Constants ────────────────────────────────────────────────────────────────
COLD_START_SCORE = 0.1
FEAT_W           = 0.6
TFIDF_W          = 0.4
CAT_BOOST        = 1.3
TOP_K_SIMILAR    = 50

FEATURES = [
    "category_name_enc",
    "brand_name_enc",
    "price_norm",
    "rating_norm",
    "engagement_score",
    "has_discount",
]
WEIGHTS = [5, 2, 2, 3, 2, 1]

EVENTS = {"view": 1, "addtocart": 3, "transaction": 5}

COMPLEMENTARY = {
    "fashion":      ["accessories", "beauty"],
    "accessories":  ["fashion", "beauty"],
    "beauty":       ["accessories", "fashion"],
    "home decor":   ["handicrafts"],
    "handicrafts":  ["home decor", "accessories"],
}


class RecommendationEngine:

    def __init__(self):
        self.df               = None
        self.feature_matrix   = None
        self.product_ids      = None
        self.similarity_cache = {}        # lazy: idx → (top_k_idx, top_k_scores)
        self.tfidf_matrix     = None
        self.tfidf_vectorizer = None
        self.encoders         = {}
        self.scaler           = None
        self.is_ready         = False

    # ─── Load & Build ─────────────────────────────────────────────────────────
    def load(self):
        self._load_data()
        self._preprocess()
        self._build_feature_matrix()
        self._build_tfidf()
        self.is_ready = True
        print("   Similarity cache: lazy (computed on first /similar request per product)")

    def _load_data(self):
        csv_path = os.getenv("PRODUCTS_CSV", "data/brandhive_products.csv")
        self.df = pd.read_csv(csv_path)
        print(f"   Products  : {len(self.df):,} rows")
        print(f"   Categories: {self.df['category_name'].unique().tolist()}")

    def _preprocess(self):
        df = self.df.copy()

        df["price"]         = pd.to_numeric(df["price"],         errors="coerce").fillna(0)
        df["discountPrice"] = pd.to_numeric(df["discountPrice"], errors="coerce").fillna(0)
        df["finalPrice"]    = pd.to_numeric(df["finalPrice"],    errors="coerce").fillna(0)

        df["viewCount"]           = pd.to_numeric(df["viewCount"],           errors="coerce").fillna(0)
        df["cartCount"]           = pd.to_numeric(df["cartCount"],           errors="coerce").fillna(0)
        df["wishlistCount"]       = pd.to_numeric(df["wishlistCount"],       errors="coerce").fillna(0)
        df["stats_averageRating"] = pd.to_numeric(df["stats_averageRating"], errors="coerce").fillna(0)
        df["stats_totalReviews"]  = pd.to_numeric(df["stats_totalReviews"],  errors="coerce").fillna(0)

        df["isOnSale"] = df["isOnSale"].astype(str).str.lower().isin(["true", "1", "yes"])
        df["isActive"] = df["isActive"].astype(str).str.lower().isin(["true", "1", "yes"])

        df = df[df["isActive"]].copy()

        df["category_name"] = df["category_name"].astype(str).str.lower().str.strip()
        df["brand_name"]    = df["brand_name"].astype(str).str.lower().str.strip()

        for col in ["category_name", "brand_name"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col])
            self.encoders[col] = le

        self.scaler = MinMaxScaler()
        df[["price_norm", "rating_norm"]] = self.scaler.fit_transform(
            df[["finalPrice", "stats_averageRating"]]
        )

        max_view = df["viewCount"].max()     + 1e-9
        max_cart = df["cartCount"].max()     + 1e-9
        max_wish = df["wishlistCount"].max() + 1e-9
        df["engagement_score"] = (
            df["viewCount"]     / max_view * 0.3 +
            df["cartCount"]     / max_cart * 0.5 +
            df["wishlistCount"] / max_wish * 0.2
        )
        df.loc[df["viewCount"] == 0, "engagement_score"] = COLD_START_SCORE

        df["has_discount"] = df["isOnSale"].astype(float)

        self.df = df.reset_index(drop=True)
        print(f"   After filter: {len(self.df):,} active products")

    def _build_feature_matrix(self):
        available = [f for f in FEATURES if f in self.df.columns]
        weights   = [WEIGHTS[FEATURES.index(f)] for f in available]
        self.feature_matrix = self.df[available].fillna(0).values * np.array(weights)
        self.product_ids    = self.df["id"].astype(str).values
        print(f"   Feature matrix : {self.feature_matrix.shape}")

    def _build_tfidf(self):
        self.df["tags_clean"] = (
            self.df["tags"].astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            .str.replace("'", "", regex=False)
            .str.replace(",", " ", regex=False)
        )
        self.df["soup"] = (
            self.df["name"].astype(str)          + " " +
            self.df["category_name"].astype(str) + " " +
            self.df["brand_name"].astype(str)    + " " +
            self.df["tags_clean"]
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", max_features=5000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df["soup"].fillna("")
        )
        print(f"   TF-IDF matrix  : {self.tfidf_matrix.shape}")

    # ─── 1. Recommend by Preferences ──────────────────────────────────────────
    def recommend_by_preferences(self, categories, min_price=0, max_price=999999,
                                  min_rating=0, brands=None, on_sale_only=False, top_n=12):
        mask = (
            self.df["category_name"].isin([c.lower().strip() for c in categories]) &
            (self.df["finalPrice"] >= min_price) &
            (self.df["finalPrice"] <= max_price) &
            (self.df["stats_averageRating"] >= min_rating)
        )
        if brands:
            mask &= self.df["brand_name"].isin([b.lower().strip() for b in brands])
        if on_sale_only:
            mask &= self.df["isOnSale"]

        filtered = self.df[mask].copy()
        if filtered.empty:
            return []

        filtered["score"] = (
            filtered["rating_norm"]      * 0.4 +
            filtered["engagement_score"] * 0.4 +
            filtered["has_discount"]     * 0.2
        )

        cols = ["id", "name", "brand_name", "category_name", "finalPrice",
                "discountPrice", "isOnSale", "discountPercentage",
                "stats_averageRating", "stats_totalReviews", "score"]
        cols = [c for c in cols if c in filtered.columns]

        return (
            filtered.sort_values("score", ascending=False)
                    .drop_duplicates(subset=["id"])
                    .head(top_n)[cols]
                    .reset_index(drop=True)
                    .to_dict(orient="records")
        )

    # ─── 2. Find Similar Products (lazy cache) ────────────────────────────────
    def find_similar(self, product_id: str, top_n=6):
        matches = np.where(self.product_ids == product_id)[0]
        if len(matches) == 0:
            return None, []

        idx     = int(matches[0])
        product = self.df.iloc[idx]

        # Compute once per product, cache the top-K
        if idx not in self.similarity_cache:
            feat_sim  = cosine_similarity([self.feature_matrix[idx]], self.feature_matrix)[0]
            tfidf_sim = cosine_similarity(self.tfidf_matrix[idx],     self.tfidf_matrix).flatten()
            combined  = feat_sim * FEAT_W + tfidf_sim * TFIDF_W
            combined[idx] = -1

            top_k = np.argpartition(combined, -TOP_K_SIMILAR)[-TOP_K_SIMILAR:]
            top_k = top_k[np.argsort(combined[top_k])[::-1]]
            self.similarity_cache[idx] = (top_k, combined[top_k].copy())

        top_idx, top_scores = self.similarity_cache[idx]
        scores = top_scores.copy().astype(float)

        same_cat = self.df.iloc[top_idx]["category_name"].values == product["category_name"]
        scores[same_cat] *= CAT_BOOST

        order   = scores.argsort()[::-1][:top_n]
        top_idx = top_idx[order]
        scores  = scores[order]

        result = self.df.iloc[top_idx][[
            "id", "name", "brand_name", "category_name",
            "finalPrice", "discountPrice", "isOnSale",
            "stats_averageRating", "stats_totalReviews"
        ]].copy()
        result["similarity_score"] = scores.round(4)

        original = {
            "id":       product["id"],
            "name":     product["name"],
            "category": product["category_name"],
            "brand":    product["brand_name"],
            "price":    float(product["finalPrice"]),
            "rating":   float(product["stats_averageRating"]),
        }
        return original, result.reset_index(drop=True).to_dict(orient="records")

    # ─── 3. Behavioral Recommendations ────────────────────────────────────────
    def behavioral_recommend(self, interactions, top_n=10):
        if not interactions:
            return (
                self.df.nlargest(top_n, "engagement_score")
                    [["id", "name", "brand_name", "category_name",
                      "finalPrice", "stats_averageRating", "engagement_score"]]
                    .reset_index(drop=True)
                    .to_dict(orient="records")
            )

        user_vec   = np.zeros(self.feature_matrix.shape[1])
        weight_sum = 0.0
        seen_ids   = set()

        for item in interactions:
            pid = str(item["product_id"])
            w   = EVENTS.get(item.get("event", "view"), 1)
            seen_ids.add(pid)

            matches = np.where(self.product_ids == pid)[0]
            if len(matches) > 0:
                idx         = int(matches[0])
                user_vec   += self.feature_matrix[idx] * w
                weight_sum += w

        if weight_sum == 0:
            return (
                self.df.nlargest(top_n, "engagement_score")
                    [["id", "name", "brand_name", "category_name",
                      "finalPrice", "stats_averageRating"]]
                    .reset_index(drop=True)
                    .to_dict(orient="records")
            )

        user_vec /= weight_sum
        scores    = cosine_similarity([user_vec], self.feature_matrix)[0]

        for pid in seen_ids:
            matches = np.where(self.product_ids == pid)[0]
            if len(matches) > 0:
                scores[int(matches[0])] = -1

        top_idx = scores.argsort()[::-1][:top_n]
        result  = self.df.iloc[top_idx][[
            "id", "name", "brand_name", "category_name",
            "finalPrice", "discountPrice", "isOnSale",
            "stats_averageRating", "stats_totalReviews"
        ]].copy()
        result["match_score"] = scores[top_idx].round(4)
        return result.reset_index(drop=True).to_dict(orient="records")

    # ─── 4. Cart Cross-Sell ────────────────────────────────────────────────────
    def cart_cross_sell(self, cart_product_ids, top_n=8):
        cart_set          = {str(p) for p in cart_product_ids}
        target_categories = set()

        for pid in cart_set:
            row = self.df[self.df["id"].astype(str) == pid]
            if not row.empty:
                cat = row.iloc[0]["category_name"]
                target_categories.update(COMPLEMENTARY.get(cat, []))

        if not target_categories:
            return []

        mask     = self.df["category_name"].isin(target_categories) & ~self.df["id"].astype(str).isin(cart_set)
        filtered = self.df[mask].copy()
        if filtered.empty:
            return []

        filtered["cs_score"] = (
            filtered["engagement_score"] * 0.50 +
            filtered["rating_norm"]       * 0.35 +
            filtered["has_discount"]      * 0.15
        )

        cols = ["id", "name", "brand_name", "category_name",
                "finalPrice", "discountPrice", "isOnSale",
                "stats_averageRating", "cs_score"]
        cols = [c for c in cols if c in filtered.columns]

        return (
            filtered.sort_values("cs_score", ascending=False)
                    .drop_duplicates("id")
                    .head(top_n)[cols]
                    .reset_index(drop=True)
                    .to_dict(orient="records")
        )

    # ─── 5. Trending Products ──────────────────────────────────────────────────
    def get_trending(self, category=None, top_n=12):
        df = self.df if category is None else self.df[self.df["category_name"] == category.lower().strip()]
        return (
            df.nlargest(top_n, "engagement_score")
              [["id", "name", "brand_name", "category_name",
                "finalPrice", "discountPrice", "isOnSale",
                "stats_averageRating", "viewCount", "cartCount", "wishlistCount"]]
              .reset_index(drop=True)
              .to_dict(orient="records")
        )

    # ─── 6. KPIs ──────────────────────────────────────────────────────────────
    def get_kpis(self):
        df = self.df
        total_views = int(df["viewCount"].sum())
        total_carts = int(df["cartCount"].sum())
        return {
            "total_products":      len(df),
            "total_views":         total_views,
            "total_cart_adds":     total_carts,
            "total_wishlists":     int(df["wishlistCount"].sum()),
            "products_on_sale":    int(df["isOnSale"].sum()),
            "avg_rating":          round(float(df["stats_averageRating"].mean()), 2),
            "avg_price_egp":       round(float(df["finalPrice"].mean()), 2),
            "conversion_estimate": round(total_carts / (total_views + 1e-9) * 100, 2),
        }

    # ─── 7. Product Stats ─────────────────────────────────────────────────────
    def get_product_stats(self):
        df = self.df
        return {
            "total_products":         len(df),
            "total_categories":       int(df["category_name"].nunique()),
            "total_brands":           int(df["brand_name"].nunique()),
            "price_range": {
                "min": float(df["finalPrice"].min()),
                "max": float(df["finalPrice"].max()),
                "avg": round(float(df["finalPrice"].mean()), 2),
            },
            "avg_rating":             round(float(df["stats_averageRating"].mean()), 2),
            "on_sale_count":          int(df["isOnSale"].sum()),
            "categories":             df["category_name"].value_counts().to_dict(),
            "top_brands":             df["brand_name"].value_counts().head(10).to_dict(),
            "avg_price_by_category":  df.groupby("category_name")["finalPrice"].mean().round(2).to_dict(),
            "avg_rating_by_category": df.groupby("category_name")["stats_averageRating"].mean().round(2).to_dict(),
        }

    # ─── 8. Helpers ───────────────────────────────────────────────────────────
    def get_categories(self):
        return sorted(self.df["category_name"].unique().tolist())

    def get_brands(self):
        return sorted(self.df["brand_name"].unique().tolist())

    def get_brands_by_category(self, category):
        return sorted(
            self.df[self.df["category_name"] == category.lower().strip()]["brand_name"].unique().tolist()
        )