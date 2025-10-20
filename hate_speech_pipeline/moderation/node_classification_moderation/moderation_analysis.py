from typing import Dict, List

import pandas as pd


class ModerationAnalyser:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def analyze_prediction_distribution(self) -> Dict:
        data = self.data

        return {
            "total_comments": len(data),
            "min_prediction": data["gcn_prediction"].min(),
            "max_prediction": data["gcn_prediction"].max(),
            "quartiles": {
                "q25": data["gcn_prediction"].quantile(0.25),
                "q50": data["gcn_prediction"].quantile(0.50),
                "q75": data["gcn_prediction"].quantile(0.75),
                "q90": data["gcn_prediction"].quantile(0.90),
                "q95": data["gcn_prediction"].quantile(0.95),
                "q99": data["gcn_prediction"].quantile(0.99),
            },
        }

    def analyze_threshold_impact(self, thresholds: List[float]) -> pd.DataFrame:
        results = []
        total_comments = len(self.data)
        data = self.data

        for threshold in thresholds:
            # Filter directly from merged_data instead of calling expensive function
            hate_comments = data[data["gcn_prediction"] >= threshold]
            moderated_count = len(hate_comments)
            moderated_percentage = (moderated_count / total_comments) * 100

            # Calculate user and subreddit impact
            affected_users = hate_comments["author"].nunique()
            affected_subreddits = hate_comments["subreddit"].nunique()

            results.append(
                {
                    "threshold": threshold,
                    "comments_moderated": moderated_count,
                    "percentage_moderated": round(moderated_percentage, 2),
                    "affected_users": affected_users,
                    "affected_subreddits": affected_subreddits,
                }
            )

        return pd.DataFrame(results)

    def analyze_user_behavior_patterns(
        self, threshold: float, user_stats: pd.DataFrame
    ) -> Dict:
        no_toxicity = len(user_stats[user_stats["hate_ratio"] == 0])
        low_toxicity = len(
            user_stats[
                (user_stats["hate_ratio"] > 0) & (user_stats["hate_ratio"] <= 0.2)
            ]
        )
        medium_toxicity = len(
            user_stats[
                (user_stats["hate_ratio"] > 0.2) & (user_stats["hate_ratio"] <= 0.5)
            ]
        )
        high_toxicity = len(
            user_stats[
                (user_stats["hate_ratio"] > 0.5) & (user_stats["hate_ratio"] <= 0.8)
            ]
        )
        extreme_toxicity = len(user_stats[user_stats["hate_ratio"] > 0.8])

        return {
            "total_users": len(user_stats),
            "behavior_categories": {
                "no_toxic_comments": no_toxicity,
                "low_toxicity_ratio": low_toxicity,
                "medium_toxicity_ratio": medium_toxicity,
                "high_toxicity_ratio": high_toxicity,
                "extreme_toxicity_ratio": extreme_toxicity,
            },
            "user_stats_summary": {
                "avg_toxic_comment_ratio": user_stats["hate_ratio"].mean(),
                "users_with_violations": len(
                    user_stats[user_stats["hate_predictions"] > 0]
                ),
            },
        }

    def analyze_subreddit_toxicity(
        self, threshold: float, subreddit_stats: pd.DataFrame
    ) -> Dict:
        clean = len(subreddit_stats[subreddit_stats["hate_ratio"] == 0])
        low_toxicity = len(
            subreddit_stats[
                (subreddit_stats["hate_ratio"] > 0)
                & (subreddit_stats["hate_ratio"] <= 0.1)
            ]
        )
        medium_toxicity = len(
            subreddit_stats[
                (subreddit_stats["hate_ratio"] > 0.1)
                & (subreddit_stats["hate_ratio"] <= 0.3)
            ]
        )
        high_toxicity = len(subreddit_stats[subreddit_stats["hate_ratio"] > 0.3])

        return {
            "total_subreddits": len(subreddit_stats),
            "toxicity_categories": {
                "clean_subreddits": clean,
                "low_toxicity": low_toxicity,
                "medium_toxicity": medium_toxicity,
                "high_toxicity": high_toxicity,
            },
            "subreddit_stats_summary": {
                "avg_toxic_comment_ratio": subreddit_stats["hate_ratio"].mean(),
                "subreddits_with_violations": len(
                    subreddit_stats[subreddit_stats["hate_predictions"] > 0]
                ),
            },
        }

    def analyze_moderation_efficiency(
        self, threshold: float, hate_comments: pd.DataFrame, bulk_result: Dict
    ) -> Dict:
        if len(hate_comments) == 0:
            return {"no_violations_found": True, "threshold_used": threshold}

        # Calculate efficiency metrics
        total_comments = len(self.data)
        precision = len(hate_comments) / total_comments if total_comments > 0 else 0
        coverage = (
            bulk_result["comments_deleted"] / len(hate_comments)
            if len(hate_comments) > 0
            else 0
        )

        return {
            "threshold_used": threshold,
            "violation_detection": {
                "violations_found": len(hate_comments),
                "detection_rate": precision * 100,
                "coverage_rate": coverage * 100,
            },
            "moderation_actions": bulk_result,
            "efficiency_metrics": {
                "comments_per_banned_user": bulk_result["comments_deleted"]
                / max(bulk_result["users_banned"], 1),
                "violation_concentration": (
                    len(hate_comments) / hate_comments["author"].nunique()
                    if len(hate_comments) > 0
                    else 0
                ),
            },
        }

    def analyze_moderation_comparison(self, threshold: float) -> Dict:
        data = self.data

        # Check if data is available
        if data is None:
            return {
                "threshold_used": threshold,
                "error": "No data available for comparison",
            }

        # Get moderation decisions based on predictions
        pred_flagged = data[data["gcn_prediction"] >= threshold]

        # Get moderation decisions based on ground truth
        truth_flagged = data[data["gcn_ground_truth"] >= threshold]

        # Find differences in moderation decisions
        pred_ids = set(pred_flagged["id"])
        truth_ids = set(truth_flagged["id"])

        # Comments flagged by predictions but not by ground truth (false positives)
        false_positive_ids = pred_ids - truth_ids
        false_positives = data[data["id"].isin(false_positive_ids)]

        # Comments flagged by ground truth but not by predictions (false negatives)
        false_negative_ids = truth_ids - pred_ids
        false_negatives = data[data["id"].isin(false_negative_ids)]

        # Comments flagged by both (true positives)
        true_positive_ids = pred_ids & truth_ids
        true_positives = data[data["id"].isin(true_positive_ids)]

        # Analyze user impact differences
        pred_users = set(pred_flagged["author"])
        truth_users = set(truth_flagged["author"])

        # Users who would be moderated differently
        over_moderated_users = (
            pred_users - truth_users
        )  # Users flagged by predictions only
        under_moderated_users = (
            truth_users - pred_users
        )  # Users flagged by ground truth only
        correctly_moderated_users = pred_users & truth_users  # Users flagged by both

        # Analyze subreddit impact differences
        pred_subreddits = set(pred_flagged["subreddit"])
        truth_subreddits = set(truth_flagged["subreddit"])

        over_moderated_subreddits = pred_subreddits - truth_subreddits
        under_moderated_subreddits = truth_subreddits - pred_subreddits
        correctly_moderated_subreddits = pred_subreddits & truth_subreddits

        # Detailed analysis of misclassified comments
        fp_analysis = {
            "count": len(false_positives),
            "users_affected": (
                len(set(false_positives["author"])) if len(false_positives) > 0 else 0
            ),
            "subreddits_affected": (
                len(set(false_positives["subreddit"]))
                if len(false_positives) > 0
                else 0
            ),
            "avg_prediction_score": (
                false_positives["gcn_prediction"].mean()
                if len(false_positives) > 0
                else 0
            ),
            "avg_ground_truth_score": (
                false_positives["gcn_ground_truth"].mean()
                if len(false_positives) > 0
                else 0
            ),
            "score_difference": (
                (
                    false_positives["gcn_prediction"]
                    - false_positives["gcn_ground_truth"]
                ).mean()
                if len(false_positives) > 0
                else 0
            ),
            "top_users": (
                false_positives["author"].value_counts().head(5).to_dict()
                if len(false_positives) > 0
                else {}
            ),
            "top_subreddits": (
                false_positives["subreddit"].value_counts().head(5).to_dict()
                if len(false_positives) > 0
                else {}
            ),
        }

        fn_analysis = {
            "count": len(false_negatives),
            "users_affected": (
                len(set(false_negatives["author"])) if len(false_negatives) > 0 else 0
            ),
            "subreddits_affected": (
                len(set(false_negatives["subreddit"]))
                if len(false_negatives) > 0
                else 0
            ),
            "avg_prediction_score": (
                false_negatives["gcn_prediction"].mean()
                if len(false_negatives) > 0
                else 0
            ),
            "avg_ground_truth_score": (
                false_negatives["gcn_ground_truth"].mean()
                if len(false_negatives) > 0
                else 0
            ),
            "score_difference": (
                (
                    false_negatives["gcn_ground_truth"]
                    - false_negatives["gcn_prediction"]
                ).mean()
                if len(false_negatives) > 0
                else 0
            ),
            "top_users": (
                false_negatives["author"].value_counts().head(5).to_dict()
                if len(false_negatives) > 0
                else {}
            ),
            "top_subreddits": (
                false_negatives["subreddit"].value_counts().head(5).to_dict()
                if len(false_negatives) > 0
                else {}
            ),
        }

        return {
            "threshold_used": threshold,
            "overall_comparison": {
                "prediction_flagged": len(pred_flagged),
                "ground_truth_flagged": len(truth_flagged),
                "correctly_flagged": len(true_positives),
                "over_moderated": len(false_positives),
                "under_moderated": len(false_negatives),
                "agreement_rate": len(true_positives)
                / max(len(pred_flagged), len(truth_flagged), 1),
            },
            "user_impact_comparison": {
                "prediction_users": len(pred_users),
                "ground_truth_users": len(truth_users),
                "correctly_moderated_users": len(correctly_moderated_users),
                "over_moderated_users": len(over_moderated_users),
                "under_moderated_users": len(under_moderated_users),
                "user_agreement_rate": len(correctly_moderated_users)
                / max(len(pred_users), len(truth_users), 1),
            },
            "subreddit_impact_comparison": {
                "prediction_subreddits": len(pred_subreddits),
                "ground_truth_subreddits": len(truth_subreddits),
                "correctly_moderated_subreddits": len(correctly_moderated_subreddits),
                "over_moderated_subreddits": len(over_moderated_subreddits),
                "under_moderated_subreddits": len(under_moderated_subreddits),
                "subreddit_agreement_rate": len(correctly_moderated_subreddits)
                / max(len(pred_subreddits), len(truth_subreddits), 1),
            },
            "false_positives_analysis": fp_analysis,
            "false_negatives_analysis": fn_analysis,
            "moderation_changes": {
                "would_not_moderate": list(over_moderated_users)[
                    :10
                ],  # Sample of users who wouldn't be moderated with ground truth
                "should_moderate": list(under_moderated_users)[
                    :10
                ],  # Sample of users who should be moderated with ground truth
                "over_moderated_subreddits": list(over_moderated_subreddits)[:10],
                "under_moderated_subreddits": list(under_moderated_subreddits)[:10],
            },
        }

    def generate_moderation_report(
        self,
        threshold: float,
        thresholds_for_impact: List[float],
        user_stats: pd.DataFrame,
        subreddit_stats: pd.DataFrame,
        hate_comments: pd.DataFrame,
        bulk_result: Dict,
    ) -> Dict:
        return {
            "analysis_threshold": threshold,
            "prediction_distribution": self.analyze_prediction_distribution(),
            "threshold_impact": self.analyze_threshold_impact(thresholds_for_impact),
            "user_behavior": self.analyze_user_behavior_patterns(threshold, user_stats),
            "subreddit_toxicity": self.analyze_subreddit_toxicity(
                threshold, subreddit_stats
            ),
            "moderation_efficiency": self.analyze_moderation_efficiency(
                threshold, hate_comments, bulk_result
            ),
            "moderation_comparison": self.analyze_moderation_comparison(threshold),
        }
