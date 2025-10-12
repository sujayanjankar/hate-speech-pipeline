"""
Things in file:
Data loading
Moderation suggestions for users/subreddits above thresholds
Moderate individual comments with auto-delete and auto-ban
Bulk moderate comments above threshold
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


class ModerationSystem:
    def __init__(self, predictions_csv_path: str, metadata_csv_path: str):
        self.predictions_csv_path = predictions_csv_path
        self.metadata_csv_path = metadata_csv_path
        self.predictions_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None

        self.load_data()

    def load_data(self):
        self.predictions_df = pd.read_csv(self.predictions_csv_path)

        metadata_cols = [
            "id",
            "parent_id",
            "link_id",
            "author",
            "subreddit",
            "created_utc",
            "body",
            "toxicity_probability_self",
            "class_self",
            "toxicity_probability_parent",
            "thread_depth",
            "score_f",
            "score_z",
            "score_bin5",
            "response_time",
            "score_parent",
            "hate_score_self",
            "hate_score_ctx",
            "user_unique_subreddits",
            "user_total_comments",
            "user_hate_comments",
            "user_hate_ratio",
            "user_avg_posting_intervall",
            "user_avg_comment_time_of_day",
            "user_hate_comments_ord",
            "user_hate_ratio_ord",
            "scorebin_0",
            "scorebin_2",
            "scorebin_3",
            "scorebin_4",
            "timestamp",
            "time_bin",
        ]

        # Read only the rows we need (same number as predictions)
        self.metadata_df = pd.read_csv(
            self.metadata_csv_path,
            usecols=metadata_cols,
            nrows=len(self.predictions_df),
        )

        # Create merged view by index (assuming both files have same order)
        self.merged_data = self.metadata_df.copy()
        self.merged_data["gcn_prediction"] = self.predictions_df["prediction"].values
        self.merged_data["gcn_ground_truth"] = self.predictions_df[
            "ground_truth"
        ].values
        self.merged_data["gcn_absolute_error"] = self.predictions_df[
            "absolute_error"
        ].values

        # Convert timestamp if needed
        if "created_utc" in self.merged_data.columns:
            self.merged_data["datetime"] = pd.to_datetime(
                self.merged_data["created_utc"], unit="s"
            )

    def get_predictions_by_threshold(self, threshold: float) -> pd.DataFrame:
        return self.merged_data[self.merged_data["gcn_prediction"] >= threshold]

    def get_user_hate_stats(self, threshold: float) -> pd.DataFrame:
        hate_comments = self.get_predictions_by_threshold(threshold)

        # Get total comments per user
        total_stats = (
            self.merged_data.groupby("author")
            .agg({"gcn_prediction": ["mean", "max"], "id": "count"})
            .round(3)
        )
        total_stats.columns = ["avg_prediction", "max_prediction", "total_comments"]

        # Get hate comments count per user
        hate_stats = hate_comments.groupby("author").size().to_frame("hate_predictions")

        # Merge and calculate hate ratio
        user_stats = total_stats.join(hate_stats, how="left").fillna(0)
        user_stats["hate_predictions"] = user_stats["hate_predictions"].astype(int)
        user_stats["hate_ratio"] = (
            user_stats["hate_predictions"] / user_stats["total_comments"]
        ).round(3)

        return user_stats.sort_values("hate_ratio", ascending=False)

    def get_subreddit_hate_stats(self, threshold: float) -> pd.DataFrame:
        hate_comments = self.get_predictions_by_threshold(threshold)

        # Get total comments per subreddit
        total_stats = (
            self.merged_data.groupby("subreddit")
            .agg({"gcn_prediction": ["mean", "max"], "id": "count"})
            .round(3)
        )
        total_stats.columns = ["avg_prediction", "max_prediction", "total_comments"]

        # Get hate comments count per subreddit
        hate_stats = (
            hate_comments.groupby("subreddit").size().to_frame("hate_predictions")
        )

        # Merge and calculate hate ratio
        subreddit_stats = total_stats.join(hate_stats, how="left").fillna(0)
        subreddit_stats["hate_predictions"] = subreddit_stats[
            "hate_predictions"
        ].astype(int)
        subreddit_stats["hate_ratio"] = (
            subreddit_stats["hate_predictions"] / subreddit_stats["total_comments"]
        ).round(3)

        return subreddit_stats.sort_values("hate_ratio", ascending=False)

    def moderate_users(
        self, user_hate_threshold: float, prediction_threshold: float
    ) -> List[str]:
        user_stats = self.get_user_hate_stats(prediction_threshold)
        users_to_moderate = user_stats[
            user_stats["hate_ratio"] >= user_hate_threshold
        ].index.tolist()

        return users_to_moderate

    def moderate_subreddits(
        self, subreddit_hate_threshold: float, prediction_threshold: float
    ) -> List[str]:
        subreddit_stats = self.get_subreddit_hate_stats(prediction_threshold)
        subreddits_to_moderate = subreddit_stats[
            subreddit_stats["hate_ratio"] >= subreddit_hate_threshold
        ].index.tolist()

        return subreddits_to_moderate

    def delete_comment(self, comment_id: str, reason: str) -> Dict:
        if self.merged_data is None:
            return {"success": False, "error": "No data loaded"}

        # Find the comment
        comment_data = self.merged_data[self.merged_data["id"] == comment_id]

        if comment_data.empty:
            return {"success": False, "error": f"Comment {comment_id} not found"}

        comment = comment_data.iloc[0]

        deletion_record = {
            "success": True,
            "comment_id": comment_id,
            "author": comment["author"],
            "subreddit": comment["subreddit"],
            "prediction_score": comment["gcn_prediction"],
            "reason": reason,
            "timestamp": datetime.now(),
            "original_text": (
                comment["body"][:100] + "..."
                if len(comment["body"]) > 100
                else comment["body"]
            ),
        }

        return deletion_record

    def ban_user(
        self, username: str, duration_days: int, reason: str, threshold: float = 0.5
    ) -> Dict:
        if self.merged_data is None:
            return {"success": False, "error": "No data loaded"}

        # Get user's comment history and stats
        user_data = self.merged_data[self.merged_data["author"] == username]

        if user_data.empty:
            return {"success": False, "error": f"User {username} not found"}

        # Calculate hate ratio with the provided threshold
        hate_comments = len(user_data[user_data["gcn_prediction"] >= threshold])
        total_comments = len(user_data)
        hate_ratio = hate_comments / total_comments if total_comments > 0 else 0

        ban_record = {
            "success": True,
            "username": username,
            "duration_days": duration_days,
            "ban_type": "permanent" if duration_days == 0 else "temporary",
            "reason": reason,
            "timestamp": datetime.now(),
            "total_comments": total_comments,
            "hate_comments": hate_comments,
            "avg_prediction": user_data["gcn_prediction"].mean(),
            "max_prediction": user_data["gcn_prediction"].max(),
            "hate_ratio": round(hate_ratio, 3),
        }

        return ban_record

    def moderate_comment(
        self,
        comment_id: str,
        threshold: float,
        auto_delete: bool,
        auto_ban_threshold: float,
    ) -> Dict:
        if self.merged_data is None:
            return {"success": False, "error": "No data loaded"}

        # Find the comment
        comment_data = self.merged_data[self.merged_data["id"] == comment_id]

        if comment_data.empty:
            return {"success": False, "error": f"Comment {comment_id} not found"}

        comment = comment_data.iloc[0]
        prediction_score = comment["gcn_prediction"]
        username = comment["author"]

        moderation_actions = {
            "comment_id": comment_id,
            "username": username,
            "subreddit": comment["subreddit"],
            "prediction_score": prediction_score,
            "actions_taken": [],
            "deleted": False,
            "banned": False,
        }

        # Check if comment exceeds hate speech threshold
        if prediction_score >= threshold:
            moderation_actions["actions_taken"].append(
                f"Hate speech detected (score: {prediction_score:.3f})"
            )

            # Auto-delete if enabled
            if auto_delete:
                delete_result = self.delete_comment(
                    comment_id, f"Hate speech (score: {prediction_score:.3f})"
                )
                if delete_result["success"]:
                    moderation_actions["deleted"] = True
                    moderation_actions["actions_taken"].append("Comment deleted")

            # Check user's overall hate ratio for potential ban
            user_stats = self.get_user_hate_stats(threshold)
            if username in user_stats.index:
                user_hate_ratio = user_stats.loc[username]["hate_ratio"]
                user_hate_count = user_stats.loc[username]["hate_predictions"]

                # Ban if user exceeds threshold and has multiple violations
                if user_hate_ratio >= auto_ban_threshold and user_hate_count >= 3:
                    ban_duration = (
                        0 if user_hate_ratio >= 0.9 else 7
                    )  # Permanent if >90% hate
                    ban_result = self.ban_user(
                        username,
                        ban_duration,
                        f"Hate ratio: {user_hate_ratio:.3f} ({user_hate_count} violations)",
                        threshold,
                    )
                    if ban_result["success"]:
                        moderation_actions["banned"] = True
                        moderation_actions["ban_duration"] = ban_duration
                        moderation_actions["actions_taken"].append(
                            f"User banned ({ban_duration} days)"
                            if ban_duration > 0
                            else "User permanently banned"
                        )
        else:
            moderation_actions["actions_taken"].append("No action required")

        return moderation_actions

    def bulk_moderate_by_threshold(
        self, threshold: float, auto_delete: bool, auto_ban_threshold: float
    ) -> Dict:
        hate_comments = self.get_predictions_by_threshold(threshold)

        results = {
            "total_comments_processed": len(hate_comments),
            "comments_deleted": 0,
            "users_banned": 0,
            "deletion_records": [],
            "ban_records": [],
            "failed_actions": [],
        }

        banned_users = set()

        for _, comment in hate_comments.iterrows():
            comment_id = comment["id"]
            username = comment["author"]

            # Moderate this comment
            moderation_result = self.moderate_comment(
                comment_id, threshold, auto_delete, auto_ban_threshold
            )

            if moderation_result.get("deleted"):
                results["comments_deleted"] += 1

            if moderation_result.get("banned") and username not in banned_users:
                results["users_banned"] += 1
                banned_users.add(username)

        return results
