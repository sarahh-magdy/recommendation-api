import os
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class RecommendationEngine:

    def __init__(self):
        self.df = None
        self.X = None
        self.pid = None           # product IDs (strings)
        self.pop_norm = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.encoders = {}
        self.scaler = None
        self.is_ready = False

    # ─── Load & Build ─────────────────────────────────────────────────
    def load(self):
        self._load_data()
        self._preprocess()
        self._build_feature_matrix()
        self._build_tfidf()
        self._build_popularity()
        self.is_ready = True

    def _load_data(self):
        csv_path = os.getenv("PRODUCTS_CSV", "data/brandhive_products.csv")
        self.df = pd.read_csv(csv_path)
        print(f"   Products : {len(self.df):,} rows")
        print(f"   Categories: {self.df['category_name'].unique().tolist()}")

    def _preprocess(self):
        df = self.df.copy()

        # نظّف الأسعار
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["discountPrice"] = pd.to_numeric(df["discountPrice"], errors="coerce")
        df["finalPrice"] = pd.to_numeric(df["finalPrice"], errors="coerce").fillna(df["price"])

        # نظّف الـ engagement
        df["viewCount"] = pd.to_numeric(df["viewCount"], errors="coerce").fillna(0)
        df["cartCount"] = pd.to_numeric(df["cartCount"], errors="coerce").fillna(0)
        df["wishlistCount"] = pd.to_numeric(df["wishlistCount"], errors="coerce").fillna(0)
        df["stats_averageRating"] = pd.to_numeric(df["stats_averageRating"], errors="coerce").fillna(0)
        df["stats_totalReviews"] = pd.to_numeric(df["stats_totalReviews"], errors="coerce").fillna(0)

        # isOnSale
        df["isOnSale"] = df["isOnSale"].astype(str).str.lower().isin(["true", "1"])
        df["isActive"] = df["isActive"].astype(str).str.lower().isin(["true", "1"])

        # فلتر المنتجات الفعّالة فقط
        df = df[df["isActive"] == True].copy()

        # Encode الـ categories والـ brands
        for col in ["category_name", "brand_name"]:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        # Normalize الأسعار والـ rating
        self.scaler = MinMaxScaler()
        df[["price_norm", "rating_norm"]] = self.scaler.fit_transform(
            df[["finalPrice", "stats_averageRating"]]
        )

        # Popularity score من engagement
        max_view = df["viewCount"].max() + 1e-9
        max_cart = df["cartCount"].max() + 1e-9
        max_wish = df["wishlistCount"].max() + 1e-9
        df["engagement_score"] = (
            df["viewCount"] / max_view * 0.3 +
            df["cartCount"] / max_cart * 0.5 +
            df["wishlistCount"] / max_wish * 0.2
        )

        # discount flag
        df["has_discount"] = df["isOnSale"].astype(float)

        self.df = df.reset_index(drop=True)
        print(f"   After filter: {len(self.df):,} active products")

    def _build_feature_matrix(self):
        FEATURES = ["category_name_enc", "brand_name_enc", "price_norm", "rating_norm", "engagement_score", "has_discount"]
        WEIGHTS  = [5, 2, 2, 3, 2, 1]

        available = [f for f in FEATURES if f in self.df.columns]
        weights   = [WEIGHTS[FEATURES.index(f)] for f in available]

        self.X   = self.df[available].fillna(0).values * np.array(weights)
        self.pid = self.df["id"].values
        print(f"   Feature matrix: {self.X.shape}")

    def _build_tfidf(self):
        # نبني soup من الاسم + category + tags
        def parse_tags(t):
            try:
                return " ".join(ast.literal_eval(t))
            except Exception:
                return str(t)

        self.df["soup"] = (
            self.df["name"].astype(str) + " " +
            self.df["category_name"].astype(str) + " " +
            self.df["brand_name"].astype(str) + " " +
            self.df["tags"].apply(parse_tags)
        )
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df["soup"].fillna(""))
        print(f"   TF-IDF matrix : {self.tfidf_matrix.shape}")

    def _build_popularity(self):
        self.pop_norm = self.df["engagement_score"].values
        print(f"   Popularity built from viewCount/cartCount/wishlistCount")

    # ─── 1. Recommend by Preferences (Category-based) ─────────────────
    def recommend_by_preferences(self, categories, min_price=0, max_price=999999,
                                  min_rating=0, brands=None, on_sale_only=False, top_n=12):
        mask = (
            self.df["category_name"].isin(categories) &
            (self.df["finalPrice"] >= min_price) &
            (self.df["finalPrice"] <= max_price) &
            (self.df["stats_averageRating"] >= min_rating)
        )
        if brands:
            mask &= self.df["brand_name"].isin(brands)
        if on_sale_only:
            mask &= self.df["isOnSale"] == True

        filtered = self.df[mask].copy()
        if filtered.empty:
            return []

        # Score = rating 40% + engagement 40% + discount bonus 20%
        filtered["score"] = (
            filtered["rating_norm"] * 0.4 +
            filtered["engagement_score"] * 0.4 +
            filtered["has_discount"] * 0.2
        )

        cols = ["id", "name", "brand_name", "category_name", "finalPrice",
                "discountPrice", "isOnSale", "discountPercentage",
                "stats_averageRating", "stats_totalReviews", "score"]
        cols = [c for c in cols if c in filtered.columns]

        result = (filtered.sort_values("score", ascending=False)
                          .drop_duplicates(subset=["name"])
                          .head(top_n)[cols]
                          .reset_index(drop=True))
        return result.to_dict(orient="records")

    # ─── 2. Find Similar Products ──────────────────────────────────────
    def find_similar(self, product_id: str, top_n=6):
        """
        بيجيب منتجات مشابهة بناءً على:
        - Cosine similarity في الـ feature space
        - TF-IDF text similarity
        - نفس الـ category بيتأثر أكتر
        """
        mask = self.pid == product_id
        if mask.sum() == 0:
            return None, []

        idx = int(np.where(mask)[0][0])
        product = self.df.iloc[idx]

        # Feature similarity
        feat_scores = cosine_similarity([self.X[idx]], self.X)[0]

        # TF-IDF similarity
        tfidf_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        # Combined score (feature 60% + tfidf 40%)
        combined = feat_scores * 0.6 + tfidf_scores * 0.4

        # boost نفس الـ category
        same_cat = self.df["category_name"] == product["category_name"]
        combined[same_cat.values] *= 1.3

        # exclude the product itself
        combined[idx] = -1

        top_idx = combined.argsort()[::-1][:top_n]
        result = self.df.iloc[top_idx][[
            "id", "name", "brand_name", "category_name",
            "finalPrice", "discountPrice", "isOnSale",
            "stats_averageRating", "stats_totalReviews"
        ]].copy()
        result["similarity_score"] = combined[top_idx].round(4)

        original = {
            "id": product["id"],
            "name": product["name"],
            "category": product["category_name"],
            "brand": product["brand_name"],
            "price": float(product["finalPrice"]),
            "rating": float(product["stats_averageRating"]),
        }
        return original, result.reset_index(drop=True).to_dict(orient="records")

    # ─── 3. Behavioral Recommendations ────────────────────────────────
    def behavioral_recommend(self, interactions, top_n=10):
        """
        بيبني user vector من تفاعلاته ويجيب أقرب منتجات ليه.
        interactions: [{"product_id": "abc123", "event": "view|addtocart|transaction"}]
        """
        EVENT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}

        if not interactions:
            # Cold start: أحسن منتجات بالـ engagement
            result = (self.df.nlargest(top_n, "engagement_score")
                        [["id", "name", "brand_name", "category_name",
                          "finalPrice", "stats_averageRating", "engagement_score"]]
                        .reset_index(drop=True))
            return result.to_dict(orient="records")

        user_vec = np.zeros(self.X.shape[1])
        weights_sum = 0
        seen_ids = set()

        for item in interactions:
            pid_val = str(item["product_id"])  # دايماً string
            weight = EVENT_WEIGHTS.get(item.get("event", "view"), 1)
            seen_ids.add(pid_val)

            mask = self.pid == pid_val
            if mask.sum() > 0:
                idx = int(np.where(mask)[0][0])
                user_vec += self.X[idx] * weight
                weights_sum += weight

        if weights_sum == 0:
            # مفيش match في الداتا — رجّع trending
            result = (self.df.nlargest(top_n, "engagement_score")
                        [["id", "name", "brand_name", "category_name",
                          "finalPrice", "stats_averageRating"]]
                        .reset_index(drop=True))
            return result.to_dict(orient="records")

        user_vec /= weights_sum
        scores = cosine_similarity([user_vec], self.X)[0]

        # exclude المنتجات اللي اتفاعل معاها
        for pid_val in seen_ids:
            mask = self.pid == pid_val
            if mask.sum() > 0:
                scores[int(np.where(mask)[0][0])] = -1

        top_idx = scores.argsort()[::-1][:top_n]
        result = self.df.iloc[top_idx][[
            "id", "name", "brand_name", "category_name",
            "finalPrice", "discountPrice", "isOnSale",
            "stats_averageRating", "stats_totalReviews"
        ]].copy()
        result["match_score"] = scores[top_idx].round(4)

        return result.reset_index(drop=True).to_dict(orient="records")

    # ─── 4. Trending Products ──────────────────────────────────────────
    def get_trending(self, category=None, top_n=12):
        df = self.df.copy()
        if category:
            df = df[df["category_name"] == category]
        result = (df.nlargest(top_n, "engagement_score")
                    [["id", "name", "brand_name", "category_name",
                      "finalPrice", "discountPrice", "isOnSale",
                      "stats_averageRating", "viewCount", "cartCount", "wishlistCount"]]
                    .reset_index(drop=True))
        return result.to_dict(orient="records")

    # ─── 5. KPIs ──────────────────────────────────────────────────────
    def get_kpis(self):
        df = self.df
        total_views    = int(df["viewCount"].sum())
        total_carts    = int(df["cartCount"].sum())
        total_wishlists = int(df["wishlistCount"].sum())
        on_sale        = int(df["isOnSale"].sum())
        avg_rating     = round(float(df["stats_averageRating"].mean()), 2)
        avg_price      = round(float(df["finalPrice"].mean()), 2)

        return {
            "total_products": len(df),
            "total_views": total_views,
            "total_cart_adds": total_carts,
            "total_wishlists": total_wishlists,
            "products_on_sale": on_sale,
            "avg_rating": avg_rating,
            "avg_price_egp": avg_price,
            "conversion_estimate": round(total_carts / (total_views + 1e-9) * 100, 2),
        }

    # ─── 6. Product Stats ─────────────────────────────────────────────
    def get_product_stats(self):
        df = self.df
        return {
            "total_products": len(df),
            "total_categories": int(df["category_name"].nunique()),
            "total_brands": int(df["brand_name"].nunique()),
            "price_range": {
                "min": float(df["finalPrice"].min()),
                "max": float(df["finalPrice"].max()),
                "avg": round(float(df["finalPrice"].mean()), 2),
            },
            "avg_rating": round(float(df["stats_averageRating"].mean()), 2),
            "on_sale_count": int(df["isOnSale"].sum()),
            "categories": df["category_name"].value_counts().to_dict(),
            "top_brands": df["brand_name"].value_counts().head(10).to_dict(),
            "avg_price_by_category": df.groupby("category_name")["finalPrice"].mean().round(2).to_dict(),
            "avg_rating_by_category": df.groupby("category_name")["stats_averageRating"].mean().round(2).to_dict(),
        }

    # ─── 7. Helpers ───────────────────────────────────────────────────
    def get_categories(self):
        return sorted(self.df["category_name"].unique().tolist())

    def get_brands(self):
        return sorted(self.df["brand_name"].unique().tolist())

    def get_brands_by_category(self, category):
        df = self.df[self.df["category_name"] == category]
        return sorted(df["brand_name"].unique().tolist())