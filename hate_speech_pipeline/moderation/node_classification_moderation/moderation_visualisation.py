import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .core_moderation import ModerationSystem


class ModerationVisualizer:
    def __init__(self, moderation_system: ModerationSystem):
        self.moderator = moderation_system

        # Create visualizations directory
        self.vis_directory = "visualisations"
        os.makedirs(self.vis_directory, exist_ok=True)

        # Set plotting style
        plt.style.use("default")

        sns.set_palette("husl")

    def save_plot(self, filename: str, dpi: int = 300):
        filepath = os.path.join(self.vis_directory, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {filepath}")

    def plot_threshold_impact(self, threshold_analysis: pd.DataFrame):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Calculate counts for flagged and auto-removed sections
        flagged_06 = (
            threshold_analysis[threshold_analysis["threshold"] == 0.6][
                "comments_moderated"
            ].iloc[0]
            if len(threshold_analysis[threshold_analysis["threshold"] == 0.6]) > 0
            else 0
        )
        auto_removed_07 = (
            threshold_analysis[threshold_analysis["threshold"] == 0.7][
                "comments_moderated"
            ].iloc[0]
            if len(threshold_analysis[threshold_analysis["threshold"] == 0.7]) > 0
            else 0
        )

        # Comments moderated vs threshold
        ax1.plot(
            threshold_analysis["threshold"],
            threshold_analysis["comments_moderated"],
            marker="o",
            linewidth=2,
            markersize=6,
            color="red",
        )
        ax1.axhline(
            y=flagged_06,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Flagged (≥0.6): {int(flagged_06):,}",
        )
        ax1.axhline(
            y=auto_removed_07,
            color="darkred",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Auto-removed (≥0.7): {int(auto_removed_07):,}",
        )
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Comments Moderated")
        ax1.set_title("Comments Moderated vs Threshold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Percentage moderated vs threshold
        flagged_06_pct = (
            threshold_analysis[threshold_analysis["threshold"] == 0.6][
                "percentage_moderated"
            ].iloc[0]
            if len(threshold_analysis[threshold_analysis["threshold"] == 0.6]) > 0
            else 0
        )
        auto_removed_07_pct = (
            threshold_analysis[threshold_analysis["threshold"] == 0.7][
                "percentage_moderated"
            ].iloc[0]
            if len(threshold_analysis[threshold_analysis["threshold"] == 0.7]) > 0
            else 0
        )

        ax2.plot(
            threshold_analysis["threshold"],
            threshold_analysis["percentage_moderated"],
            marker="s",
            linewidth=2,
            markersize=6,
            color="orange",
        )
        ax2.axhline(
            y=flagged_06_pct,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Flagged (≥0.6): {flagged_06_pct:.2f}%",
        )
        ax2.axhline(
            y=auto_removed_07_pct,
            color="darkred",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Auto-removed (≥0.7): {auto_removed_07_pct:.2f}%",
        )
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Percentage Moderated (%)")
        ax2.set_title("Percentage of Comments Moderated vs Threshold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Users affected vs threshold
        ax3.plot(
            threshold_analysis["threshold"],
            threshold_analysis["affected_users"],
            marker="^",
            linewidth=2,
            markersize=6,
            color="blue",
        )
        ax3.set_xlabel("Threshold")
        ax3.set_ylabel("Users Affected")
        ax3.set_title("Users Affected vs Threshold")
        ax3.grid(True, alpha=0.3)

        # Subreddits affected vs threshold
        ax4.plot(
            threshold_analysis["threshold"],
            threshold_analysis["affected_subreddits"],
            marker="d",
            linewidth=2,
            markersize=6,
            color="green",
        )
        ax4.set_xlabel("Threshold")
        ax4.set_ylabel("Subreddits Affected")
        ax4.set_title("Subreddits Affected vs Threshold")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot("threshold_impact_analysis.png")

    def plot_subreddit_toxicity(self, threshold: float, subreddit_analysis: Dict):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Pie chart of toxicity categories
        categories = subreddit_analysis["toxicity_categories"]
        labels = [label.replace("_", " ").title() for label in categories.keys()]
        sizes = list(categories.values())
        colors = [
            "#ffcccc",
            "#ff9999",
            "#ff6666",
            "#ff3333",
        ]  # Light to dark red gradient

        # Calculate percentages for better label handling
        total = sum(sizes)
        percentages = [(size / total) * 100 if total > 0 else 0 for size in sizes]

        # Custom autopct function to only show percentages >= 2%
        def autopct_format(pct):
            return f"{pct:.1f}%" if pct >= 2.0 else ""

        # Create pie chart with improved label positioning
        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=None,  # We'll add labels separately
            autopct=autopct_format,
            colors=colors,
            startangle=90,
            pctdistance=0.85,  # Move percentages closer to edge
            explode=[0.05 if p < 5.0 else 0 for p in percentages],
        )  # Explode small slices

        # Add legend instead of direct labels to avoid overlap
        legend_labels = [
            f"{label}: {size} ({pct:.1f}%)"
            for label, size, pct in zip(labels, sizes, percentages)
        ]
        ax1.legend(
            wedges,
            legend_labels,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )

        # Improve autotext formatting
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

        ax1.set_title(
            f"Subreddit Toxicity Categories (Threshold: {threshold})",
            fontsize=12,
            pad=20,
        )

        # Horizontal bar chart with value labels
        bars = ax2.barh(
            range(len(labels)), sizes, color=colors, alpha=0.8, edgecolor="black"
        )
        ax2.set_ylabel("Toxicity Category")
        ax2.set_xlabel("Number of Subreddits")
        ax2.set_title(f"Subreddit Toxicity Distribution (Threshold: {threshold})")
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels)
        ax2.grid(True, alpha=0.3, axis="x")

        # Add value labels on bars
        for i, (bar, size, pct) in enumerate(zip(bars, sizes, percentages)):
            width = bar.get_width()
            ax2.text(
                width + max(sizes) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{size} ({pct:.1f}%)",
                ha="left",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()
        self.save_plot("subreddit_toxicity_analysis.png")

    def plot_top_users_subreddits(
        self,
        threshold: float,
        user_stats: pd.DataFrame,
        subreddit_stats: pd.DataFrame,
        top_n: int = 10,
    ):
        # Filter for meaningful data
        user_stats_filtered = user_stats[user_stats["total_comments"] > 3].head(top_n)
        subreddit_stats_filtered = subreddit_stats[
            subreddit_stats["total_comments"] > 10
        ].head(top_n)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Top users by hate ratio
        if len(user_stats_filtered) > 0:
            y_pos = np.arange(len(user_stats_filtered))
            ax1.barh(
                y_pos,
                user_stats_filtered["hate_ratio"],
                color="lightcoral",
                alpha=0.8,
                edgecolor="black",
            )
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(user_stats_filtered.index)
            ax1.set_xlabel("Hate Ratio")
            ax1.set_title(
                f"Top {len(user_stats_filtered)} Users by Hate Ratio (>3 comments, threshold: {threshold})"
            )
            ax1.grid(True, alpha=0.3, axis="x")

        # Top subreddits by hate ratio
        if len(subreddit_stats_filtered) > 0:
            y_pos = np.arange(len(subreddit_stats_filtered))
            ax2.barh(
                y_pos,
                subreddit_stats_filtered["hate_ratio"],
                color="lightblue",
                alpha=0.8,
                edgecolor="black",
            )
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(subreddit_stats_filtered.index)
            ax2.set_xlabel("Hate Ratio")
            ax2.set_title(
                f"Top {len(subreddit_stats_filtered)} Subreddits by Hate Ratio (>10 comments, threshold: {threshold})"
            )
            ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        self.save_plot("top_users_subreddits.png")

    def plot_moderation_comparison(self, threshold: float, comparison: Dict):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Overall comparison bar chart
        overall = comparison["overall_comparison"]
        categories = [
            "Prediction\nFlagged",
            "Ground Truth\nFlagged",
            "Correctly\nFlagged",
            "Over-moderated\n(False Pos)",
            "Under-moderated\n(False Neg)",
        ]
        values = [
            overall["prediction_flagged"],
            overall["ground_truth_flagged"],
            overall["correctly_flagged"],
            overall["over_moderated"],
            overall["under_moderated"],
        ]
        colors = ["blue", "green", "purple", "red", "orange"]

        bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")
        ax1.set_ylabel("Number of Comments")
        ax1.set_title(f"Moderation Comparison Overview (Threshold: {threshold})")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(values) * 0.01,
                f"{value}",
                ha="center",
                va="bottom",
            )

        # Agreement rates pie chart
        agreement_data = [
            overall["correctly_flagged"],
            overall["over_moderated"] + overall["under_moderated"],
        ]
        agreement_labels = ["Agreement", "Disagreement"]
        colors_pie = ["lightgreen", "lightcoral"]

        ax2.pie(
            agreement_data,
            labels=agreement_labels,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        ax2.set_title(
            f"Prediction vs Ground Truth Agreement\n(Comments, Threshold: {threshold})"
        )

        plt.tight_layout()
        self.save_plot("moderation_comparison_analysis.png")

    def generate_all_visualizations(
        self,
        threshold: float,
        user_stats: pd.DataFrame,
        subreddit_stats: pd.DataFrame,
        subreddit_analysis: Dict,
        comparison: Dict,
        threshold_analysis=None,
    ):

        if threshold_analysis is not None:
            self.plot_threshold_impact(threshold_analysis)

        self.plot_subreddit_toxicity(threshold, subreddit_analysis)
        self.plot_top_users_subreddits(threshold, user_stats, subreddit_stats)
        self.plot_moderation_comparison(threshold, comparison)

        print(f"All visualizations saved to '{self.vis_directory}' directory!")
