from datetime import datetime
from typing import Dict

import pandas as pd

from .diffusion_moderation import DiffusionModerationSystem


class DiffusionAnalysis:
    def __init__(self, diffusion_system: DiffusionModerationSystem):
        self.diffusion_system = diffusion_system
        self.data = diffusion_system.data
        self.thread_analysis = diffusion_system.thread_analysis
        self.actionable_threads = diffusion_system.actionable_threads

    def get_dataset_overview(self) -> Dict:
        data = self.data

        # Basic statistics
        total_comments = len(data)
        hate_downstream_count = len(data[data["has_hate_downstream"] == 1])
        high_toxicity_count = len(data[data["toxicity_probability_self"] >= 0.7])
        medium_toxicity_count = len(data[data["toxicity_probability_self"] >= 0.5])

        # Subreddit and user diversity
        unique_subreddits = data["subreddit"].nunique()
        unique_authors = data["author"].nunique() if "author" in data.columns else 0

        # Thread structure statistics
        total_threads_analyzed = len(self.thread_analysis)
        actionable_thread_count = len(self.actionable_threads)

        # Toxicity distribution
        toxicity_stats = {
            "mean": data["toxicity_probability_self"].mean(),
            "median": data["toxicity_probability_self"].median(),
            "std": data["toxicity_probability_self"].std(),
            "min": data["toxicity_probability_self"].min(),
            "max": data["toxicity_probability_self"].max(),
            "q25": data["toxicity_probability_self"].quantile(0.25),
            "q75": data["toxicity_probability_self"].quantile(0.75),
            "q90": data["toxicity_probability_self"].quantile(0.90),
            "q95": data["toxicity_probability_self"].quantile(0.95),
        }

        return {
            "dataset_summary": {
                "total_comments": total_comments,
                "hate_downstream_comments": hate_downstream_count,
                "hate_downstream_percentage": round(
                    (hate_downstream_count / total_comments) * 100, 2
                ),
                "high_toxicity_comments": high_toxicity_count,
                "medium_toxicity_comments": medium_toxicity_count,
                "unique_subreddits": unique_subreddits,
                "unique_authors": unique_authors,
            },
            "thread_structure": {
                "total_threads_analyzed": total_threads_analyzed,
                "actionable_threads": actionable_thread_count,
                "actionable_percentage": (
                    round((actionable_thread_count / total_threads_analyzed) * 100, 2)
                    if total_threads_analyzed > 0
                    else 0
                ),
            },
            "toxicity_distribution": toxicity_stats,
            "analysis_timestamp": datetime.now(),
        }

    def analyze_subreddit_patterns(self) -> Dict:
        data = self.data

        # Group by subreddit and calculate hate downstream statistics
        subreddit_stats = (
            data.groupby("subreddit")
            .agg(
                {
                    "has_hate_downstream": ["sum", "count", "mean"],
                    "toxicity_probability_self": ["mean", "max", "std"],
                }
            )
            .round(3)
        )

        # Flatten column names
        subreddit_stats.columns = [
            "hate_downstream_count",
            "total_comments",
            "hate_downstream_ratio",
            "avg_toxicity",
            "max_toxicity",
            "toxicity_std",
        ]

        # Add derived metrics
        subreddit_stats["risk_score"] = (
            subreddit_stats["hate_downstream_ratio"] * 0.6
            + subreddit_stats["avg_toxicity"] * 0.4
        ).round(3)

        # Sort by risk score
        subreddit_stats = subreddit_stats.sort_values("risk_score", ascending=False)

        # Calculate actionable threads per subreddit
        actionable_by_subreddit = {}
        for thread in self.actionable_threads:
            subreddit = thread["parent_subreddit"]
            actionable_by_subreddit[subreddit] = (
                actionable_by_subreddit.get(subreddit, 0) + 1
            )

        # Add actionable thread counts
        subreddit_stats["actionable_threads"] = subreddit_stats.index.map(
            lambda x: actionable_by_subreddit.get(x, 0)
        )

        return {
            "subreddit_analysis": subreddit_stats,
            "risk_summary": {
                "high_risk_subreddits": len(
                    subreddit_stats[subreddit_stats["risk_score"] >= 0.5]
                ),
                "medium_risk_subreddits": len(
                    subreddit_stats[
                        (subreddit_stats["risk_score"] >= 0.3)
                        & (subreddit_stats["risk_score"] < 0.5)
                    ]
                ),
                "low_risk_subreddits": len(
                    subreddit_stats[subreddit_stats["risk_score"] < 0.3]
                ),
                "highest_risk_subreddit": (
                    subreddit_stats.index[0] if len(subreddit_stats) > 0 else None
                ),
                "highest_risk_score": (
                    subreddit_stats["risk_score"].iloc[0]
                    if len(subreddit_stats) > 0
                    else 0
                ),
            },
        }

    def get_thread_summary(self) -> Dict:
        if not self.thread_analysis:
            return {"error": "No thread analysis available"}

        # Thread statistics
        thread_stats = []
        escalation_count = 0
        high_toxicity_threads = 0

        for thread_id, stats in self.thread_analysis.items():
            thread_stats.append(
                {
                    "thread_id": thread_id,
                    "child_count": stats["child_count"],
                    "avg_toxicity": stats["avg_toxicity"],
                    "max_toxicity": stats["max_toxicity"],
                    "parent_toxicity": stats["parent_toxicity"],
                    "toxicity_escalation": stats["toxicity_escalation"],
                    "actionable": stats["actionable"],
                }
            )

            if stats["toxicity_escalation"]:
                escalation_count += 1
            if stats["avg_toxicity"] >= 0.7:
                high_toxicity_threads += 1

        thread_df = pd.DataFrame(thread_stats)

        if len(thread_df) == 0:
            return {"error": "No thread data to analyze"}

        # Calculate summary statistics
        summary = {
            "thread_overview": {
                "total_threads": len(thread_df),
                "escalation_threads": escalation_count,
                "high_toxicity_threads": high_toxicity_threads,
                "actionable_threads": len(thread_df[thread_df["actionable"]]),
            },
            "toxicity_patterns": {
                "avg_thread_toxicity": thread_df["avg_toxicity"].mean(),
                "max_thread_toxicity": thread_df["max_toxicity"].max(),
                "escalation_rate": round((escalation_count / len(thread_df)) * 100, 2),
                "actionable_rate": round(
                    (len(thread_df[thread_df["actionable"]]) / len(thread_df)) * 100, 2
                ),
            },
            "thread_size_distribution": {
                "avg_children_per_thread": thread_df["child_count"].mean(),
                "max_children_in_thread": thread_df["child_count"].max(),
            },
        }

        return summary

    def get_complete_summary(self) -> Dict:
        dataset_overview = self.get_dataset_overview()
        subreddit_analysis = self.analyze_subreddit_patterns()
        thread_summary = self.get_thread_summary()

        # Apply moderation actions for additional insights
        moderation_actions = self.diffusion_system.apply_tiered_moderation()

        complete_summary = {
            "analysis_metadata": {
                "analysis_type": "Diffusion-based Hate Speech Moderation",
                "timestamp": datetime.now(),
                "data_source": self.diffusion_system.diffusion_data_path,
                "thresholds": {
                    "high_toxicity": self.diffusion_system.high_toxicity_threshold,
                    "medium_toxicity": self.diffusion_system.medium_toxicity_threshold,
                    "low_toxicity": self.diffusion_system.low_toxicity_threshold,
                    "thread_actionable": self.diffusion_system.thread_actionable_threshold,
                },
            },
            "dataset_overview": dataset_overview,
            "subreddit_patterns": subreddit_analysis,
            "thread_analysis": thread_summary,
            "moderation_impact": {
                "immediate_removals": len(moderation_actions["immediate_removals"]),
                "immediate_review": len(moderation_actions["immediate_review"]),
                "flag_for_review": len(moderation_actions["flag_for_review"]),
                "monitor_escalation": len(moderation_actions["monitor_escalation"]),
                "total_actions": sum(
                    len(actions) for actions in moderation_actions.values()
                ),
            },
        }

        return complete_summary

    def classify_subreddit_risk(self) -> pd.DataFrame:
        subreddit_patterns = self.analyze_subreddit_patterns()
        subreddit_stats = subreddit_patterns["subreddit_analysis"].copy()

        # Define risk categories
        def categorize_risk(row):
            risk_score = row["risk_score"]
            hate_ratio = row["hate_downstream_ratio"]
            avg_toxicity = row["avg_toxicity"]

            if risk_score >= 0.7 or (hate_ratio >= 0.5 and avg_toxicity >= 0.6):
                return "CRITICAL"
            elif risk_score >= 0.5 or (hate_ratio >= 0.3 and avg_toxicity >= 0.5):
                return "HIGH"
            elif risk_score >= 0.3 or (hate_ratio >= 0.1 and avg_toxicity >= 0.4):
                return "MEDIUM"
            elif risk_score >= 0.1:
                return "LOW"
            else:
                return "MINIMAL"

        subreddit_stats["risk_category"] = subreddit_stats.apply(
            categorize_risk, axis=1
        )

        # Add recommendation
        def get_recommendation(risk_category, actionable_threads, total_comments):
            if risk_category == "CRITICAL":
                return "Immediate intervention required - Enhanced monitoring"
            elif risk_category == "HIGH":
                return "Close monitoring - Frequent moderation checks"
            elif risk_category == "MEDIUM":
                return "Regular monitoring - Automated flagging"
            elif risk_category == "LOW":
                return "Periodic review - Standard moderation"
            else:
                return "Minimal oversight required"

        subreddit_stats["recommendation"] = subreddit_stats.apply(
            lambda row: get_recommendation(
                row["risk_category"], row["actionable_threads"], row["total_comments"]
            ),
            axis=1,
        )

        return subreddit_stats

    def generate_detailed_thread_report(self) -> pd.DataFrame:
        thread_details = []

        for thread_id, stats in self.thread_analysis.items():
            parent_comment = stats["parent_comment"]

            # Get child comment details if they exist
            child_comments = self.data[self.data["parent_id"] == thread_id]

            thread_record = {
                "thread_id": thread_id,
                "parent_subreddit": parent_comment["subreddit"],
                "parent_author": parent_comment.get("author", "Unknown"),
                "parent_toxicity": stats["parent_toxicity"],
                "parent_has_hate_downstream": parent_comment["has_hate_downstream"],
                "child_count": stats["child_count"],
                "avg_child_toxicity": stats["avg_toxicity"],
                "max_child_toxicity": stats["max_toxicity"],
                "min_child_toxicity": stats["min_toxicity"],
                "toxicity_std": stats["std_toxicity"],
                "toxicity_escalation": stats["toxicity_escalation"],
                "is_actionable": stats["actionable"],
                "risk_level": self._calculate_thread_risk_level(stats),
                "moderation_priority": self._get_moderation_priority(stats),
                "parent_created_utc": parent_comment.get("created_utc", 0),
            }

            # Add child comment statistics
            if len(child_comments) > 0:
                thread_record.update(
                    {
                        "child_authors_count": (
                            child_comments["author"].nunique()
                            if "author" in child_comments.columns
                            else 0
                        ),
                        "child_avg_score": child_comments.get(
                            "score_f", pd.Series([0])
                        ).mean(),
                        "thread_depth_avg": child_comments.get(
                            "thread_depth", pd.Series([1])
                        ).mean(),
                    }
                )
            else:
                thread_record.update(
                    {
                        "child_authors_count": 0,
                        "child_avg_score": 0,
                        "thread_depth_avg": 1,
                    }
                )

            thread_details.append(thread_record)

        df = pd.DataFrame(thread_details)

        # Sort by risk level and toxicity
        risk_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "MINIMAL": 0}
        df["risk_numeric"] = df["risk_level"].map(risk_order)
        df = df.sort_values(
            ["risk_numeric", "avg_child_toxicity"], ascending=[False, False]
        )
        df = df.drop("risk_numeric", axis=1)

        return df

    def _calculate_thread_risk_level(self, thread_stats: Dict) -> str:
        avg_toxicity = thread_stats["avg_toxicity"]
        max_toxicity = thread_stats["max_toxicity"]
        child_count = thread_stats["child_count"]
        escalation = thread_stats["toxicity_escalation"]

        # Multi-factor risk assessment
        risk_score = 0

        # Toxicity scoring
        if avg_toxicity >= 0.7:
            risk_score += 3
        elif avg_toxicity >= 0.6:
            risk_score += 2
        elif avg_toxicity >= 0.5:
            risk_score += 1

        # Max toxicity bonus
        if max_toxicity >= 0.8:
            risk_score += 2
        elif max_toxicity >= 0.6:
            risk_score += 1

        # Thread size factor
        if child_count >= 10:
            risk_score += 2
        elif child_count >= 5:
            risk_score += 1

        # Escalation penalty
        if escalation:
            risk_score += 1

        # Determine risk level
        if risk_score >= 6:
            return "CRITICAL"
        elif risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        elif risk_score >= 1:
            return "LOW"
        else:
            return "MINIMAL"

    def _get_moderation_priority(self, thread_stats: Dict) -> str:
        if thread_stats["actionable"]:
            if thread_stats["avg_toxicity"] >= 0.7:
                return "IMMEDIATE"
            elif thread_stats["toxicity_escalation"]:
                return "URGENT"
            else:
                return "HIGH"
        elif thread_stats["avg_toxicity"] >= 0.5:
            return "MODERATE"
        else:
            return "LOW"
