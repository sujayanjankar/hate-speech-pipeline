import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from .diffusion_analysis import DiffusionAnalysis
from .diffusion_moderation import DiffusionModerationSystem

class DiffusionModerationVisualizer:
    def __init__(self, diffusion_analysis: DiffusionAnalysis):
        self.analysis = diffusion_analysis
        self.diffusion_system = diffusion_analysis.diffusion_system
        
        # Create visualizations directory
        self.vis_directory = os.path.join("visualisations")
        os.makedirs(self.vis_directory, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
    def save_plot(self, filename: str, dpi: int = 300):
        filepath = os.path.join(self.vis_directory, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {filepath}")
        return filepath
    
    def plot_thread_analysis(self):
        thread_summary = self.analysis.get_thread_summary()
        
        if 'error' in thread_summary:
            print(f"Cannot create thread analysis plot: {thread_summary['error']}")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Thread-Level Diffusion Analysis', fontsize=16, fontweight='bold')
        
        # 1. Thread Type Distribution
        thread_overview = thread_summary['thread_overview']
        categories = ['Total Threads', 'Escalation Threads', 'High Toxicity', 'Actionable']
        counts = [
            thread_overview['total_threads'],
            thread_overview['escalation_threads'],
            thread_overview['high_toxicity_threads'],
            thread_overview['actionable_threads']
        ]
        colors = ['lightblue', 'orange', 'red', 'darkred']
        
        bars = ax1.bar(categories, counts, color=colors, alpha=0.7)
        ax1.set_ylabel('Thread Count')
        ax1.set_title('Thread Classification Overview')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Toxicity Patterns
        toxicity_patterns = thread_summary['toxicity_patterns']
        pattern_labels = ['Avg Thread\nToxicity', 'Max Thread\nToxicity']
        pattern_values = [
            toxicity_patterns['avg_thread_toxicity'],
            toxicity_patterns['max_thread_toxicity']
        ]
        
        bars = ax2.bar(pattern_labels, pattern_values, color=['skyblue', 'red'], alpha=0.7)
        ax2.set_ylabel('Toxicity Score')
        ax2.set_title('Thread Toxicity Patterns')
        ax2.axhline(y=0.7, color='orange', linestyle='--', label='High Toxicity Threshold')
        ax2.axhline(y=0.5, color='yellow', linestyle='--', label='Medium Toxicity Threshold')
        ax2.legend()
        
        # Add value labels
        for bar, value in zip(bars, pattern_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Escalation and Actionable Rates
        rates = [
            toxicity_patterns['escalation_rate'],
            toxicity_patterns['actionable_rate']
        ]
        rate_labels = ['Escalation Rate (%)', 'Actionable Rate (%)']
        colors = ['orange', 'red']
        
        bars = ax3.bar(rate_labels, rates, color=colors, alpha=0.7)
        ax3.set_ylabel('Percentage')
        ax3.set_title('Thread Risk Indicators')
        ax3.set_ylim(0, 100)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Thread Size Distribution
        size_distribution = thread_summary['thread_size_distribution']
        if 'avg_children_per_thread' in size_distribution:
            size_labels = ['Avg Children\nper Thread', 'Max Children\nin Thread']
            size_values = [
                size_distribution['avg_children_per_thread'],
                size_distribution['max_children_in_thread']
            ]
            
            bars = ax4.bar(size_labels, size_values, color=['lightgreen', 'darkgreen'], alpha=0.7)
            ax4.set_ylabel('Number of Children')
            ax4.set_title('Thread Size Distribution')
            
            # Add value labels
            for bar, value in zip(bars, size_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(size_values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Thread size data unavailable', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        return self.save_plot('diffusion_thread_analysis.png')
    
    def plot_moderation_impact(self):
        complete_summary = self.analysis.get_complete_summary()
        moderation_impact = complete_summary['moderation_impact']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Diffusion Moderation Impact Analysis', fontsize=16, fontweight='bold')
        
        # 1. Moderation Actions Distribution
        action_types = ['Immediate\nRemovals', 'Immediate\nReview', 'Flag for\nReview', 'Monitor\nEscalation']
        action_counts = [
            moderation_impact['immediate_removals'],
            moderation_impact['immediate_review'],
            moderation_impact['flag_for_review'],
            moderation_impact['monitor_escalation']
        ]
        colors = ['darkred', 'red', 'orange', 'yellow']
        
        bars = ax1.bar(action_types, action_counts, color=colors, alpha=0.7)
        ax1.set_ylabel('Action Count')
        ax1.set_title(f'Moderation Actions Distribution\n(Total: {moderation_impact["total_actions"]:,} actions)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, action_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(action_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Action Distribution Pie Chart
        non_zero_actions = [(action, count) for action, count in zip(action_types, action_counts) if count > 0]
        if non_zero_actions:
            labels, counts = zip(*non_zero_actions)
            colors_pie = [colors[action_types.index(label)] for label in labels]
            
            # Custom function to only show percentage if slice is large enough
            def autopct_format(pct):
                return f'{pct:.1f}%' if pct > 3 else ''
            
            wedges, texts, autotexts = ax2.pie(counts, 
                                              labels=None,  # Remove direct labels to avoid overlap
                                              colors=colors_pie, 
                                              autopct=autopct_format, 
                                              startangle=90,
                                              pctdistance=0.85)
            
            # Create custom legend with counts
            total_actions = sum(counts)
            legend_labels = [f'{label}: {count} ({count/total_actions*100:.1f}%)' 
                            for label, count in zip(labels, counts)]
            ax2.legend(wedges, legend_labels, title="Action Types", 
                      loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            ax2.set_title('Moderation Actions Percentage')
            
            # Improve text styling
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
        else:
            ax2.text(0.5, 0.5, 'No moderation actions required', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Moderation Actions Percentage')
        
        # 3. Threshold Impact Comparison
        thresholds = complete_summary['analysis_metadata']['thresholds']
        threshold_labels = ['High Toxicity', 'Medium Toxicity', 'Low Toxicity', 'Thread Actionable']
        threshold_values = [
            thresholds['high_toxicity'],
            thresholds['medium_toxicity'],
            thresholds['low_toxicity'],
            thresholds['thread_actionable']
        ]
        
        bars = ax3.bar(threshold_labels, threshold_values, color='lightcoral', alpha=0.7)
        ax3.set_ylabel('Threshold Value')
        ax3.set_title('Moderation Thresholds Configuration')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, threshold_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Dataset Coverage Analysis
        dataset_overview = complete_summary['dataset_overview']['dataset_summary']
        coverage_labels = ['Total Comments', 'Hate Downstream', 'High Toxicity', 'Medium Toxicity']
        coverage_counts = [
            dataset_overview['total_comments'],
            dataset_overview['hate_downstream_comments'],
            dataset_overview['high_toxicity_comments'],
            dataset_overview['medium_toxicity_comments']
        ]
        
        bars = ax4.bar(coverage_labels, coverage_counts, color=['lightblue', 'red', 'orange', 'yellow'], alpha=0.7)
        ax4.set_ylabel('Comment Count')
        ax4.set_title('Dataset Coverage Analysis')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, coverage_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(coverage_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return self.save_plot('diffusion_moderation_impact.png')
    
    def plot_comprehensive_risk_matrix(self):
        subreddit_stats = self.analysis.classify_subreddit_risk()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Comprehensive Risk Assessment Matrix', fontsize=16, fontweight='bold')
        
        # 1. Risk Score vs Hate Downstream Ratio Scatter
        risk_colors = {
            'CRITICAL': 'darkred',
            'HIGH': 'red', 
            'MEDIUM': 'orange',
            'LOW': 'yellow',
            'MINIMAL': 'green'
        }
        
        for risk_cat in subreddit_stats['risk_category'].unique():
            mask = subreddit_stats['risk_category'] == risk_cat
            subset = subreddit_stats[mask]
            ax1.scatter(subset['hate_downstream_ratio'], subset['risk_score'], 
                       c=risk_colors[risk_cat], label=risk_cat, s=100, alpha=0.7)
        
        ax1.set_xlabel('Hate Downstream Ratio')
        ax1.set_ylabel('Risk Score')
        ax1.set_title('Risk Score vs Hate Diffusion')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk Category Distribution
        risk_counts = subreddit_stats['risk_category'].value_counts()
        colors = [risk_colors[cat] for cat in risk_counts.index]
        
        # Custom function to only show percentage if slice is large enough
        def autopct_format(pct):
            return f'{pct:.1f}%' if pct > 5 else ''
        
        # Create pie chart with improved label handling
        wedges, texts, autotexts = ax2.pie(risk_counts.values, 
                                          labels=None,  # Remove direct labels to avoid overlap
                                          colors=colors, 
                                          autopct=autopct_format, 
                                          startangle=90,
                                          pctdistance=0.85)  # Move percentages closer to center
        
        # Create custom legend with counts
        legend_labels = [f'{cat}: {count} ({count/risk_counts.sum()*100:.1f}%)' 
                        for cat, count in risk_counts.items()]
        ax2.legend(wedges, legend_labels, title="Risk Categories", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        ax2.set_title('Subreddit Risk Category Distribution')
        
        # Improve text styling for better readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        return self.save_plot('diffusion_comprehensive_risk_matrix.png')
    
    def create_all_visualizations(self):
        print("Creating diffusion moderation visualizations...")
        
        plots_created = []
        
        try:
            # Thread analysis
            plot_path = self.plot_thread_analysis()
            if plot_path:
                plots_created.append(plot_path)
            
            # Moderation impact
            plot_path = self.plot_moderation_impact()
            plots_created.append(plot_path)
            
            # Comprehensive risk matrix (simplified)
            plot_path = self.plot_comprehensive_risk_matrix()
            plots_created.append(plot_path)
            
            print(f"\nSuccessfully created {len(plots_created)} visualizations:")
            for plot in plots_created:
                print(f"  - {plot}")
                
            return plots_created
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            return plots_created