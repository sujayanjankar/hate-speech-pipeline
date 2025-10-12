import os
import sys

import moderation_display as display
import pandas as pd
from diffusion_classification_moderation.diffusion_analysis import DiffusionAnalysis
from diffusion_classification_moderation.diffusion_moderation import (
    DiffusionModerationSystem,
)
from diffusion_classification_moderation.diffusion_moderation_visualisation import (
    DiffusionModerationVisualizer,
)
from node_classification_moderation.core_moderation import ModerationSystem
from node_classification_moderation.moderation_analysis import ModerationAnalyser
from node_classification_moderation.moderation_visualisation import ModerationVisualizer

# Analysis thresholds
MAIN_ANALYSIS_THRESHOLD = 0.6
THRESHOLD_IMPACT_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Moderation action thresholds
USER_HATE_THRESHOLD = 0.5
SUBREDDIT_HATE_THRESHOLD = 0.1
AUTO_DELETE_THRESHOLD = 0.6
AUTO_BAN_THRESHOLD = 0.4

# Display filtering thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.9
BASIC_TOXIC_THRESHOLD = 0.6
MIN_USER_COMMENTS = 3
MIN_SUBREDDIT_COMMENTS = 10

# Bulk moderation settings
BULK_AUTO_DELETE = True
BULK_AUTO_BAN_THRESHOLD = 0.4

# Ban duration settings (days, 0 = permanent)
DEFAULT_BAN_DURATION = 7
PERMANENT_BAN_THRESHOLD = 0.9


def compute_all_analysis_data(moderator, analyser):
    display.show_computing_status()

    # Basic data
    basic_data = {
        "total_comments": (
            len(moderator.merged_data) if moderator.merged_data is not None else 0
        ),
        "high_confidence_comments": len(
            moderator.get_predictions_by_threshold(HIGH_CONFIDENCE_THRESHOLD)
        ),
        "toxic_comments": len(
            moderator.get_predictions_by_threshold(BASIC_TOXIC_THRESHOLD)
        ),
    }

    # User and subreddit stats
    user_stats = moderator.get_user_hate_stats(MAIN_ANALYSIS_THRESHOLD)
    subreddit_stats = moderator.get_subreddit_hate_stats(MAIN_ANALYSIS_THRESHOLD)

    # Moderation recommendations
    users_to_moderate = moderator.moderate_users(
        user_hate_threshold=USER_HATE_THRESHOLD,
        prediction_threshold=MAIN_ANALYSIS_THRESHOLD,
    )
    subreddits_to_moderate = moderator.moderate_subreddits(
        subreddit_hate_threshold=SUBREDDIT_HATE_THRESHOLD,
        prediction_threshold=MAIN_ANALYSIS_THRESHOLD,
    )

    # Bulk moderation demo data
    bulk_demo_data = None
    threshold_used = None
    for threshold in [i / 10 for i in range(9, 0, -1)]:
        demo_comments = moderator.get_predictions_by_threshold(threshold)
        if len(demo_comments) > 0:
            threshold_used = threshold
            bulk_demo_data = moderator.bulk_moderate_by_threshold(
                threshold=threshold_used,
                auto_delete=BULK_AUTO_DELETE,
                auto_ban_threshold=BULK_AUTO_BAN_THRESHOLD,
            )
            bulk_demo_data["threshold_used"] = threshold_used
            bulk_demo_data["comment_count"] = len(demo_comments)
            break

    # Analysis data
    prediction_analysis = analyser.analyze_prediction_distribution()
    threshold_analysis = analyser.analyze_threshold_impact(THRESHOLD_IMPACT_RANGE)
    user_behavior_analysis = analyser.analyze_user_behavior_patterns(
        MAIN_ANALYSIS_THRESHOLD, user_stats
    )
    subreddit_analysis = analyser.analyze_subreddit_toxicity(
        MAIN_ANALYSIS_THRESHOLD, subreddit_stats
    )

    # Get hate comments for efficiency analysis
    hate_comments = (
        moderator.merged_data[
            moderator.merged_data["gcn_prediction"] >= MAIN_ANALYSIS_THRESHOLD
        ]
        if moderator.merged_data is not None
        else pd.DataFrame()
    )
    efficiency_analysis = analyser.analyze_moderation_efficiency(
        MAIN_ANALYSIS_THRESHOLD, hate_comments, bulk_demo_data
    )
    comparison = analyser.analyze_moderation_comparison(MAIN_ANALYSIS_THRESHOLD)

    # Return all computed data
    return {
        "basic_data": basic_data,
        "user_stats": user_stats,
        "subreddit_stats": subreddit_stats,
        "users_to_moderate": users_to_moderate,
        "subreddits_to_moderate": subreddits_to_moderate,
        "bulk_demo_data": bulk_demo_data,
        "prediction_analysis": prediction_analysis,
        "threshold_analysis": threshold_analysis,
        "user_behavior_analysis": user_behavior_analysis,
        "subreddit_analysis": subreddit_analysis,
        "efficiency_analysis": efficiency_analysis,
        "comparison": comparison,
    }


# Function to generate all visualisations
def generate_all_visualisations(visualizer, analysis_data):
    display.show_visualisation_status()

    # Generate static moderation visualisations
    visualizer.generate_all_visualizations(
        MAIN_ANALYSIS_THRESHOLD,
        analysis_data["user_stats"],
        analysis_data["subreddit_stats"],
        analysis_data["subreddit_analysis"],
        analysis_data["comparison"],
        analysis_data["threshold_analysis"],
    )


def run_diffusion_demo(diffusion_data_path):
    display.show_diffusion_system_header()

    # Initialize the diffusion moderation system
    display.show_diffusion_initialisation_status()
    try:
        # Add path to import diffusion modules
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Check if file exists
        if not os.path.exists(diffusion_data_path):
            display.show_diffusion_data_error(diffusion_data_path)
            return

        diffusion_system = DiffusionModerationSystem(diffusion_data_path)
        display.show_diffusion_data_loaded(
            diffusion_data_path, len(diffusion_system.data)
        )

    except Exception as e:
        display.show_diffusion_system_error(str(e))
        return

    # Initialize analysis system
    display.show_analysis_initialisation_status()
    try:
        analysis = DiffusionAnalysis(diffusion_system)
        overview = analysis.get_dataset_overview()
        display.show_analysis_initialised(overview)

    except Exception as e:
        display.show_analysis_error(str(e))
        return

    # Initialize visualisation system
    display.show_visualiser_initialisation_status()
    try:
        visualizer = DiffusionModerationVisualizer(analysis)
        display.show_visualiser_initialised(visualizer.vis_directory)

    except Exception as e:
        display.show_visualiser_error(str(e))
        return

    # Create visualisations
    display.show_creating_visualisations_status()

    try:
        plots_created = visualizer.create_all_visualizations()

        if plots_created:
            display.show_visualisations_created(plots_created, visualizer.vis_directory)
        else:
            display.show_no_visualisations_created()

    except Exception as e:
        display.show_visualisation_creation_error(str(e))
        import traceback

        traceback.print_exc()

    # Print summary analysis
    display.show_analysis_summary_status()

    try:
        complete_summary = analysis.get_complete_summary()
        display.show_complete_analysis_summary(complete_summary)

    except Exception as e:
        display.show_summary_error(str(e))

    display.show_diffusion_complete()


def run_node_classification_demo(predictions_csv, metadata_csv):
    display.show_loading_status(predictions_csv, metadata_csv)

    # Initialize systems
    moderator = ModerationSystem(predictions_csv, metadata_csv)
    if moderator.merged_data is None:
        display.show_error_message("Failed to load data")
        return

    analyser = ModerationAnalyser(moderator.merged_data)
    visualizer = ModerationVisualizer(moderator)

    # Compute and display analysis
    analysis_data = compute_all_analysis_data(moderator, analyser)
    display.display_all_results(analysis_data)
    generate_all_visualisations(visualizer, analysis_data)


# Main function
def demo():
    predictions_csv = "../gcn_predictions.csv"
    metadata_csv = "../data/retrain_test10_with_embeddings.csv"
    diffusion_data_path = "../retrain_test10_with_has_hate_downstream.csv"

    # Run the node classification demoW
    run_node_classification_demo(predictions_csv, metadata_csv)

    # Run the diffusion demo
    run_diffusion_demo(diffusion_data_path)


if __name__ == "__main__":
    demo()
