import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set


class DiffusionModerationSystem:
    def __init__(self, diffusion_data_path: str = "retrain_test10_with_has_hate_downstream.csv"):
        self.diffusion_data_path = diffusion_data_path
        
        # Toxicity thresholds for tiered moderation
        self.high_toxicity_threshold = 0.8    # High toxicity
        self.medium_toxicity_threshold = 0.6  # Medium toxicity
        self.low_toxicity_threshold = 0.4     # Low toxicity
        self.thread_actionable_threshold = 0.5
        
        # Load and process data
        self.data = self.load_data()
        
        # Pre-computed data that gets reused
        self.hate_downstream_comments = self.data[self.data['has_hate_downstream'] == 1].copy()

        self.no_hate_high_toxicity_comments = self.data[
            (self.data['has_hate_downstream'] == 0) & 
            (self.data['toxicity_probability_self'] >= self.high_toxicity_threshold)
        ].copy()
        
        # Thread analysis and statistics
        self.thread_analysis = self.analyze_thread_structures()
        self.actionable_threads = self.get_actionable_threads()
        
        # Moderation tracking
        self.moderation_actions: List[Dict] = []
        self.blocked_comments: Set[str] = set()
    
    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.diffusion_data_path)
        
        # Convert timestamp if available
        if 'created_utc' in data.columns:
            data['datetime'] = pd.to_datetime(data['created_utc'], unit='s')
        
        return data.sort_values('created_utc') if 'created_utc' in data.columns else data
    
    def analyze_thread_structures(self) -> Dict:
        print(f"Analyzing {len(self.hate_downstream_comments)} comments with hate downstream...")
        
        thread_stats = {}
        
        for _, parent_comment in self.hate_downstream_comments.iterrows():
            parent_id = parent_comment['id']
            child_comments = self.data[self.data['parent_id'] == parent_id]
            
            if len(child_comments) > 0:
                child_toxicities = child_comments['toxicity_probability_self']
                avg_toxicity = child_toxicities.mean()
                
                thread_stats[parent_id] = {
                    'parent_comment': parent_comment,
                    'child_count': len(child_comments),
                    'avg_toxicity': avg_toxicity,
                    'max_toxicity': child_toxicities.max(),
                    'min_toxicity': child_toxicities.min(),
                    'std_toxicity': child_toxicities.std(),
                    'parent_toxicity': parent_comment['toxicity_probability_self'],
                    'actionable': avg_toxicity >= self.thread_actionable_threshold,
                    'toxicity_escalation': avg_toxicity > parent_comment['toxicity_probability_self']
                }
            else:
                thread_stats[parent_id] = {
                    'parent_comment': parent_comment,
                    'child_count': 0,
                    'avg_toxicity': 0,
                    'max_toxicity': 0,
                    'min_toxicity': 0,
                    'std_toxicity': 0,
                    'parent_toxicity': parent_comment['toxicity_probability_self'],
                    'actionable': False,
                    'toxicity_escalation': False
                }
        
        return thread_stats
    
    def get_actionable_threads(self) -> List[Dict]:
        actionable = []
        
        for thread_id, thread_data in self.thread_analysis.items():
            if thread_data['actionable'] and thread_data['child_count'] > 0:
                actionable.append({
                    'thread_id': thread_id,
                    'parent_subreddit': thread_data['parent_comment']['subreddit'],
                    'parent_toxicity': thread_data['parent_toxicity'],
                    'child_count': thread_data['child_count'],
                    'avg_toxicity': thread_data['avg_toxicity'],
                    'max_toxicity': thread_data['max_toxicity'],
                    'min_toxicity': thread_data['min_toxicity'],
                    'toxicity_escalation': thread_data['toxicity_escalation']
                })
        
        return sorted(actionable, key=lambda x: x['avg_toxicity'], reverse=True)
    
    
    def apply_tiered_moderation(self) -> Dict[str, List]:
        actions = {
            'immediate_removals': [],
            'immediate_review': [],
            'flag_for_review': [],
            'monitor_escalation': []
        }
        
        self.moderation_actions.clear()  # Reset previous actions
        
        # Process hate downstream comments
        for _, comment in self.hate_downstream_comments.iterrows():
            action = self.determine_moderation_action(comment, has_hate_downstream=True)
            actions[action['type']].append(action)
            self.moderation_actions.append(action)
        
        # Process high toxicity comments without hate downstream
        for _, comment in self.no_hate_high_toxicity_comments.iterrows():
            action = self.determine_moderation_action(comment, has_hate_downstream=False)
            actions[action['type']].append(action)
            self.moderation_actions.append(action)
        
        return actions
    
    def determine_moderation_action(self, comment: pd.Series, has_hate_downstream: bool) -> Dict:
        toxicity = comment['toxicity_probability_self']
        thread_data = self.thread_analysis.get(comment['id'], {})
        avg_thread_toxicity = thread_data.get('avg_toxicity', 0)
        
        base_action = {
            'comment_id': comment['id'],
            'subreddit': comment['subreddit'],
            'parent_id': comment['parent_id'],
            'toxicity': toxicity,
            'avg_thread_toxicity': avg_thread_toxicity,
            'has_hate_downstream': has_hate_downstream,
            'timestamp': datetime.now(),
            'created_utc': comment['created_utc']
        }
        
        if has_hate_downstream:
            # High individual toxicity OR high thread toxicity = immediate removal
            if toxicity >= self.high_toxicity_threshold or avg_thread_toxicity >= self.high_toxicity_threshold:
                priority = 'CRITICAL' if avg_thread_toxicity >= self.high_toxicity_threshold else 'HIGH'
                return {**base_action,
                       'type': 'immediate_removals',
                       'action': 'remove_and_lock',
                       'reason': f'Toxicity: {toxicity:.3f}, Thread avg: {avg_thread_toxicity:.3f} - with confirmed hate downstream',
                       'priority': priority}
            
            # Medium individual toxicity OR medium thread toxicity = immediate review
            elif toxicity >= self.medium_toxicity_threshold or avg_thread_toxicity >= self.medium_toxicity_threshold:
                return {**base_action,
                       'type': 'immediate_review',
                       'action': 'flag_immediate_review_monitor',
                       'reason': f'Toxicity: {toxicity:.3f}, Thread avg: {avg_thread_toxicity:.3f} - with confirmed hate downstream',
                       'priority': 'HIGH'}
            
            # Low toxicity = flag for review
            else:
                priority = 'MEDIUM' if toxicity >= self.low_toxicity_threshold else 'LOW'
                return {**base_action,
                       'type': 'flag_for_review',
                       'action': 'flag_for_review',
                       'reason': f'Toxicity: {toxicity:.3f}, Thread avg: {avg_thread_toxicity:.3f} - with confirmed hate downstream',
                       'priority': priority}
        else:
            # No hate downstream but high toxicity - monitor for escalation
            priority = 'HIGH' if avg_thread_toxicity >= self.medium_toxicity_threshold else 'MEDIUM'
            return {**base_action,
                   'type': 'monitor_escalation',
                   'action': 'monitor_hate_escalation',
                   'reason': f'Toxicity: {toxicity:.3f}, Thread avg: {avg_thread_toxicity:.3f} - monitor for escalation',
                   'priority': priority}

