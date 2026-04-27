"""
Interaction Simulator
=====================
Generates realistic synthetic user-item interactions for training
collaborative filtering models when real user data is unavailable.

Design decisions:
- Users are assigned category preferences (2-4 categories each) to mimic
  real browsing behavior where users have topical interests.
- Interaction types have different weights:
    view=1, click=2, purchase=5
  This is standard in implicit feedback systems (see Hu et al. 2008).
- Timestamps span 90 days to enable recency-based features.
- Purchase probability decays with price to simulate price sensitivity.
"""

import numpy as np
import pandas as pd
from typing import Optional
import datetime


class InteractionSimulator:
    """Generates synthetic user-item interaction data."""

    INTERACTION_WEIGHTS = {"view": 1.0, "click": 2.0, "purchase": 5.0}

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        df: pd.DataFrame,
        n_users: int = 500,
        interactions_per_user: tuple = (10, 80),
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic interactions.

        Args:
            df: Product DataFrame (must have 'asin', 'categoryName', 'price',
                'stars', 'popularity_score').
            n_users: Number of synthetic users to create.
            interactions_per_user: (min, max) interactions per user.
            save_path: If provided, saves the DataFrame to CSV.

        Returns:
            DataFrame with columns:
                [user_id, item_idx, asin, interaction_type, weight, timestamp]
        """
        print(f"🧪 Simulating interactions for {n_users} users...")

        categories = df["categoryName"].unique()
        n_items = len(df)

        # Pre-compute category-to-item mapping for fast lookup
        cat_to_items = {}
        for cat in categories:
            cat_to_items[cat] = df.index[df["categoryName"] == cat].tolist()

        # Pre-compute item popularity weights (used for sampling)
        pop_scores = df["popularity_score"].values.copy()
        pop_scores = np.clip(pop_scores, 0.01, None)  # avoid zero

        records = []
        base_time = datetime.datetime(2025, 1, 1)

        for user_id in range(n_users):
            # Each user has 2-4 preferred categories
            n_prefs = self.rng.randint(2, min(5, len(categories) + 1))
            preferred_cats = self.rng.choice(categories, size=n_prefs, replace=False)

            # Pool: 70% from preferred categories, 30% random (exploration)
            pref_items = []
            for cat in preferred_cats:
                pref_items.extend(cat_to_items.get(cat, []))

            n_interactions = self.rng.randint(
                interactions_per_user[0], interactions_per_user[1] + 1
            )

            for _ in range(n_interactions):
                # 70% chance to pick from preferred categories
                if self.rng.random() < 0.70 and pref_items:
                    item_idx = int(self.rng.choice(pref_items))
                else:
                    # Popularity-weighted random sampling
                    probs = pop_scores / pop_scores.sum()
                    item_idx = int(self.rng.choice(n_items, p=probs))

                # Determine interaction type based on item quality
                item_stars = float(df.iloc[item_idx].get("stars", 3.0))
                item_price = float(df.iloc[item_idx].get("price", 10.0))

                # Higher-rated, lower-priced items more likely to be purchased
                purchase_prob = (item_stars / 5.0) * (1.0 / (1.0 + item_price / 100.0))
                purchase_prob = np.clip(purchase_prob * 0.3, 0.02, 0.25)

                roll = self.rng.random()
                if roll < purchase_prob:
                    itype = "purchase"
                elif roll < purchase_prob + 0.3:
                    itype = "click"
                else:
                    itype = "view"

                # Random timestamp within 90-day window
                delta = datetime.timedelta(
                    days=self.rng.randint(0, 90),
                    hours=self.rng.randint(0, 24),
                    minutes=self.rng.randint(0, 60),
                )
                ts = base_time + delta

                records.append(
                    {
                        "user_id": user_id,
                        "item_idx": item_idx,
                        "asin": df.iloc[item_idx]["asin"],
                        "interaction_type": itype,
                        "weight": self.INTERACTION_WEIGHTS[itype],
                        "timestamp": ts,
                    }
                )

        interactions_df = pd.DataFrame(records)

        # Summary stats
        n_total = len(interactions_df)
        n_views = (interactions_df["interaction_type"] == "view").sum()
        n_clicks = (interactions_df["interaction_type"] == "click").sum()
        n_purchases = (interactions_df["interaction_type"] == "purchase").sum()
        print(f"✅ Generated {n_total} interactions:")
        print(f"   Views: {n_views} | Clicks: {n_clicks} | Purchases: {n_purchases}")
        print(
            f"   Users: {n_users} | Items touched: {interactions_df['item_idx'].nunique()}"
        )

        if save_path:
            interactions_df.to_csv(save_path, index=False)
            print(f"💾 Saved to {save_path}")

        return interactions_df
