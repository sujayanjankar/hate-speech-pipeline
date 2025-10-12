import os

# Display filtering thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.9
BASIC_TOXIC_THRESHOLD = 0.6
MIN_USER_COMMENTS = 3
MIN_SUBREDDIT_COMMENTS = 10
USER_HATE_THRESHOLD = 0.5
SUBREDDIT_HATE_THRESHOLD = 0.1
MAIN_ANALYSIS_THRESHOLD = 0.6


# Moderation display functions
def show_basic_statistics(basic_data):
    print("\n=== BASIC STATISTICS ===")
    print(f"Total comments analyzed: {basic_data['total_comments']}")
    print(
        f"Comments predicted as hate speech (>{HIGH_CONFIDENCE_THRESHOLD} probability): {basic_data['high_confidence_comments']}"
    )
    print(
        f"Comments predicted as toxic (>{BASIC_TOXIC_THRESHOLD} probability): {basic_data['toxic_comments']}"
    )


def show_top_users(user_stats):
    print(f"\n=== TOP USERS BY HATE RATIO (>{MIN_USER_COMMENTS} comments) ===")
    user_stats_filtered = user_stats[user_stats["total_comments"] > MIN_USER_COMMENTS]
    print(user_stats_filtered.head(10))


def show_top_subreddits(subreddit_stats):
    print(
        f"\n=== TOP SUBREDDITS BY HATE RATIO (>{MIN_SUBREDDIT_COMMENTS} comments) ==="
    )
    subreddit_stats_filtered = subreddit_stats[
        subreddit_stats["total_comments"] > MIN_SUBREDDIT_COMMENTS
    ]
    print(subreddit_stats_filtered.head(10))


def show_moderation_recommendations(users_to_moderate, subreddits_to_moderate):
    print("\n=== MODERATION RECOMMENDATIONS ===")
    if users_to_moderate:
        print(
            f"Users recommended for moderation (hate ratio >{USER_HATE_THRESHOLD}): {users_to_moderate[:5]}..."
        )
    else:
        print(f"No users exceed the hate threshold of {USER_HATE_THRESHOLD}")

    if subreddits_to_moderate:
        print(
            f"Subreddits recommended for moderation (hate ratio >{SUBREDDIT_HATE_THRESHOLD}): {subreddits_to_moderate[:5]}..."
        )
    else:
        print(f"No subreddits exceed the hate threshold of {SUBREDDIT_HATE_THRESHOLD}")


def show_comment_moderation_demo(bulk_demo_data):
    print("\n=== COMMENT-BASED MODERATION DEMO ===")
    if bulk_demo_data:
        print(
            f"Found {bulk_demo_data['comment_count']} comments (>{bulk_demo_data['threshold_used']} threshold)"
        )
        print("Processing...")
        print(f"Processed: {bulk_demo_data['total_comments_processed']} comments")
        print(f"Deleted: {bulk_demo_data['comments_deleted']} comments")
        print(f"Banned: {bulk_demo_data['users_banned']} users")
    else:
        print("No comments found for demo")


# Analysis display functions
def show_prediction_analysis(dist_analysis):
    print("\n=== PREDICTION DISTRIBUTION ANALYSIS ===")
    print(f"Total comments analyzed: {dist_analysis['total_comments']}")
    print(
        f"Score range: {dist_analysis['min_prediction']:.3f} - {dist_analysis['max_prediction']:.3f}"
    )

    print("\nScore Percentiles:")
    for percentile, value in dist_analysis["quartiles"].items():
        print(f"  {percentile.upper()}: {value:.3f}")


def show_threshold_analysis(threshold_analysis):
    print("\n=== THRESHOLD IMPACT ANALYSIS ===")
    print(threshold_analysis.to_string(index=False))


def show_user_behavior_analysis(user_analysis):
    print(f"\n=== USER BEHAVIOR ANALYSIS (threshold: {MAIN_ANALYSIS_THRESHOLD}) ===")
    print(f"Total users analyzed: {user_analysis['total_users']}")
    print(
        f"Users with violations: {user_analysis['user_stats_summary']['users_with_violations']}"
    )
    print(
        f"Average toxic comment ratio: {user_analysis['user_stats_summary']['avg_toxic_comment_ratio']:.3f}"
    )

    print("\nBehavior Categories:")
    for category, count in user_analysis["behavior_categories"].items():
        percentage = (count / user_analysis["total_users"]) * 100
        print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")


def show_subreddit_analysis(subreddit_analysis):
    print(
        f"\n=== SUBREDDIT TOXICITY ANALYSIS (threshold: {MAIN_ANALYSIS_THRESHOLD}) ==="
    )
    print(f"Total subreddits analyzed: {subreddit_analysis['total_subreddits']}")
    print(
        f"Subreddits with violations: {subreddit_analysis['subreddit_stats_summary']['subreddits_with_violations']}"
    )
    print(
        f"Average toxic comment ratio: {subreddit_analysis['subreddit_stats_summary']['avg_toxic_comment_ratio']:.3f}"
    )

    print("\nToxicity Categories:")
    for category, count in subreddit_analysis["toxicity_categories"].items():
        percentage = (count / subreddit_analysis["total_subreddits"]) * 100
        print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")


def show_efficiency_analysis(efficiency_analysis):
    print(
        f"\n=== MODERATION EFFICIENCY ANALYSIS (threshold: {MAIN_ANALYSIS_THRESHOLD}) ==="
    )

    if efficiency_analysis.get("no_violations_found"):
        print(
            f"No violations found at threshold {efficiency_analysis['threshold_used']}"
        )
        return

    print(f"Analysis threshold: {efficiency_analysis['threshold_used']}")
    print(
        f"Violations detected: {efficiency_analysis['violation_detection']['violations_found']}"
    )
    print(
        f"Detection rate: {efficiency_analysis['violation_detection']['detection_rate']:.2f}%"
    )
    print(
        f"Coverage rate: {efficiency_analysis['violation_detection']['coverage_rate']:.2f}%"
    )

    actions = efficiency_analysis["moderation_actions"]
    print(f"\nModeration Actions:")
    print(f"  Comments processed: {actions['total_comments_processed']}")
    print(f"  Comments deleted: {actions['comments_deleted']}")
    print(f"  Users banned: {actions['users_banned']}")

    metrics = efficiency_analysis["efficiency_metrics"]
    print(f"\nEfficiency Metrics:")
    print(f"  Comments per banned user: {metrics['comments_per_banned_user']:.1f}")
    print(f"  Violation concentration: {metrics['violation_concentration']:.1f}")


# Ground truth comparison display functions
def show_moderation_comparison(comparison):
    print(
        f"\n=== MODERATION COMPARISON: PREDICTIONS vs GROUND TRUTH (threshold: {MAIN_ANALYSIS_THRESHOLD}) ==="
    )
    print(f"Analysis threshold: {comparison['threshold_used']}")

    # Overall comparison
    overall = comparison["overall_comparison"]
    print(f"\nOverall Moderation Comparison:")
    print(f"  Comments flagged by predictions: {overall['prediction_flagged']}")
    print(f"  Comments flagged by ground truth: {overall['ground_truth_flagged']}")
    print(f"  Correctly flagged (agreement): {overall['correctly_flagged']}")
    print(f"  Over-moderated (false positives): {overall['over_moderated']}")
    print(f"  Under-moderated (false negatives): {overall['under_moderated']}")
    print(f"  Agreement rate: {overall['agreement_rate']:.2%}")

    # User impact comparison
    user_impact = comparison["user_impact_comparison"]
    print(f"\nUser Impact Comparison:")
    print(f"  Users affected by predictions: {user_impact['prediction_users']}")
    print(f"  Users affected by ground truth: {user_impact['ground_truth_users']}")
    print(f"  Correctly identified users: {user_impact['correctly_moderated_users']}")
    print(f"  Over-moderated users: {user_impact['over_moderated_users']}")
    print(f"  Under-moderated users: {user_impact['under_moderated_users']}")
    print(f"  User agreement rate: {user_impact['user_agreement_rate']:.2%}")

    # Subreddit impact comparison
    subreddit_impact = comparison["subreddit_impact_comparison"]
    print(f"\nSubreddit Impact Comparison:")
    print(
        f"  Subreddits affected by predictions: {subreddit_impact['prediction_subreddits']}"
    )
    print(
        f"  Subreddits affected by ground truth: {subreddit_impact['ground_truth_subreddits']}"
    )
    print(
        f"  Correctly identified subreddits: {subreddit_impact['correctly_moderated_subreddits']}"
    )
    print(
        f"  Over-moderated subreddits: {subreddit_impact['over_moderated_subreddits']}"
    )
    print(
        f"  Under-moderated subreddits: {subreddit_impact['under_moderated_subreddits']}"
    )
    print(
        f"  Subreddit agreement rate: {subreddit_impact['subreddit_agreement_rate']:.2%}"
    )

    # False positives analysis
    fp = comparison["false_positives_analysis"]
    print(f"\nFalse Positives Analysis (Over-moderation):")
    print(f"  Count: {fp['count']}")
    if fp["count"] > 0:
        print(f"  Users affected: {fp['users_affected']}")
        print(f"  Subreddits affected: {fp['subreddits_affected']}")
        print(f"  Average prediction score: {fp['avg_prediction_score']:.3f}")
        print(f"  Average ground truth score: {fp['avg_ground_truth_score']:.3f}")
        print(f"  Average score difference: {fp['score_difference']:.3f}")
        if fp["top_users"]:
            print(f"  Top over-moderated users: {list(fp['top_users'].keys())[:3]}")
        if fp["top_subreddits"]:
            print(
                f"  Top over-moderated subreddits: {list(fp['top_subreddits'].keys())[:3]}"
            )

    # False negatives analysis
    fn = comparison["false_negatives_analysis"]
    print(f"\nFalse Negatives Analysis (Under-moderation):")
    print(f"  Count: {fn['count']}")
    if fn["count"] > 0:
        print(f"  Users affected: {fn['users_affected']}")
        print(f"  Subreddits affected: {fn['subreddits_affected']}")
        print(f"  Average prediction score: {fn['avg_prediction_score']:.3f}")
        print(f"  Average ground truth score: {fn['avg_ground_truth_score']:.3f}")
        print(f"  Average score difference: {fn['score_difference']:.3f}")
        if fn["top_users"]:
            print(f"  Top under-moderated users: {list(fn['top_users'].keys())[:3]}")
        if fn["top_subreddits"]:
            print(
                f"  Top under-moderated subreddits: {list(fn['top_subreddits'].keys())[:3]}"
            )

    # Moderation changes
    changes = comparison["moderation_changes"]
    print(f"\nSpecific Moderation Changes:")
    if changes["would_not_moderate"]:
        print(
            f"  Users who wouldn't be moderated with ground truth: {changes['would_not_moderate']}"
        )
    if changes["should_moderate"]:
        print(
            f"  Users who should be moderated with ground truth: {changes['should_moderate']}"
        )
    if changes["over_moderated_subreddits"]:
        print(f"  Over-moderated subreddits: {changes['over_moderated_subreddits']}")
    if changes["under_moderated_subreddits"]:
        print(f"  Under-moderated subreddits: {changes['under_moderated_subreddits']}")


# Status message display functions
def show_system_status(message):
    print(f"\n{message}")


def show_loading_status(predictions_csv, metadata_csv):
    print("=== Hate Speech Moderation System Demo ===")
    print(f"Loading predictions from: {predictions_csv}")
    print(f"Loading metadata from: {metadata_csv}")


def show_computing_status():
    print("\n=== COMPUTING ALL ANALYSIS DATA ===")
    print("This may take a moment...")


def show_visualisation_status():
    print("\n=== GENERATING VISUALISATIONS ===")
    print("Generating static moderation visualisations...")


def show_error_message(message):
    print(f"Error: {message}")


# Diffusion system display functions
def show_diffusion_system_header():
    print("=== DIFFUSION MODERATION ANALYSIS SYSTEM ===")


def show_diffusion_initialisation_status():
    print("\n1. Initializing Diffusion Moderation System...")


def show_diffusion_data_loaded(data_path, data_size):
    print(f"Successfully loaded diffusion data from {data_path}")
    print(f"Dataset size: {data_size:,} comments")


def show_diffusion_data_error(data_path):
    print(f"Error: Data file not found at {data_path}")


def show_diffusion_system_error(error_msg):
    print(f"Error initializing diffusion system: {str(error_msg)}")


def show_analysis_initialisation_status():
    print("\n2. Initializing Analysis System...")


def show_analysis_initialised(overview):
    print("Analysis system initialized successfully")
    print(
        f"  Comments with hate downstream: {overview['dataset_summary']['hate_downstream_comments']:,}"
    )
    print(
        f"  Threads analyzed: {overview['thread_structure']['total_threads_analyzed']:,}"
    )
    print(
        f"  Actionable threads: {overview['thread_structure']['actionable_threads']:,}"
    )


def show_analysis_error(error_msg):
    print(f"Error initializing analysis: {str(error_msg)}")


def show_visualiser_initialisation_status():
    print("\n3. Initializing Visualisation System...")


def show_visualiser_initialised(vis_directory):
    print("Visualisation system initialized successfully")
    print(f"Output directory: {vis_directory}")


def show_visualiser_error(error_msg):
    print(f"Error initializing visualiser: {str(error_msg)}")


def show_creating_visualisations_status():
    print("\n4. Creating Visualisations...")


def show_visualisations_created(plots_created, vis_directory):
    print(f"\nSuccessfully created {len(plots_created)} visualisation plots!")
    print("\nGenerated visualisations:")
    for i, plot_path in enumerate(plots_created, 1):
        plot_name = os.path.basename(plot_path)
        print(f"  {i}. {plot_name}")
    print(f"\nAll visualisations saved to: {vis_directory}")


def show_no_visualisations_created():
    print("No visualisations were created successfully")


def show_visualisation_creation_error(error_msg):
    print(f"Error creating visualisations: {str(error_msg)}")


def show_analysis_summary_status():
    print("\n5. Analysis Summary...")


def show_complete_analysis_summary(complete_summary):
    # Dataset summary
    dataset_summary = complete_summary["dataset_overview"]["dataset_summary"]
    print(f"Dataset Overview:")
    print(f"  Total comments: {dataset_summary['total_comments']:,}")
    print(
        f"  Hate downstream rate: {dataset_summary['hate_downstream_percentage']:.1f}%"
    )
    print(f"  Unique subreddits: {dataset_summary['unique_subreddits']:,}")

    # Thread analysis
    thread_structure = complete_summary["dataset_overview"]["thread_structure"]
    print(f"\nThread Analysis:")
    print(f"  Total threads: {thread_structure['total_threads_analyzed']:,}")
    print(
        f"  Actionable threads: {thread_structure['actionable_threads']:,} ({thread_structure['actionable_percentage']:.1f}%)"
    )

    # Moderation impact
    moderation_impact = complete_summary["moderation_impact"]
    print(f"\nModeration Impact:")
    print(f"  Immediate removals: {moderation_impact['immediate_removals']:,}")
    print(f"  Immediate review: {moderation_impact['immediate_review']:,}")
    print(f"  Flag for review: {moderation_impact['flag_for_review']:,}")
    print(f"  Monitor escalation: {moderation_impact['monitor_escalation']:,}")
    print(f"  Total actions: {moderation_impact['total_actions']:,}")

    # Toxicity distribution summary
    toxicity_stats = complete_summary["dataset_overview"]["toxicity_distribution"]
    print(f"\nToxicity Distribution:")
    print(f"  Mean toxicity: {toxicity_stats['mean']:.3f}")
    print(f"  95th percentile: {toxicity_stats['q95']:.3f}")
    print(f"  Maximum toxicity: {toxicity_stats['max']:.3f}")


def show_summary_error(error_msg):
    print(f"Error generating summary: {str(error_msg)}")


def show_diffusion_complete():
    print("DIFFUSION ANALYSIS COMPLETE")


# Main display orchestrator function
def display_all_results(analysis_data):
    print("\n>>>>>DISPLAYING MODERATION RESULTS<<<<<")

    show_basic_statistics(analysis_data["basic_data"])
    show_top_users(analysis_data["user_stats"])
    show_top_subreddits(analysis_data["subreddit_stats"])
    show_moderation_recommendations(
        analysis_data["users_to_moderate"], analysis_data["subreddits_to_moderate"]
    )
    show_comment_moderation_demo(analysis_data["bulk_demo_data"])

    print("\n>>>>>DISPLAYING ANALYSIS RESULTS<<<<<")

    show_prediction_analysis(analysis_data["prediction_analysis"])
    show_threshold_analysis(analysis_data["threshold_analysis"])
    show_user_behavior_analysis(analysis_data["user_behavior_analysis"])
    show_subreddit_analysis(analysis_data["subreddit_analysis"])
    show_efficiency_analysis(analysis_data["efficiency_analysis"])

    print(
        "\n>>>>>DISPLAYING MODERATION COMPARISON RESULTS (Ground Truth vs Predictions)<<<<<"
    )
    show_moderation_comparison(analysis_data["comparison"])
