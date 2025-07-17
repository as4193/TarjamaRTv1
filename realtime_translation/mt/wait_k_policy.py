import time
import numpy as np
from typing import List, Dict, Optional, Iterator, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Handle config import with fallback
try:
    from ..config import get_config
except ImportError:
    # Fallback for when running as standalone module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from config import get_config
    except ImportError:
        # Create a mock config for testing
        def get_config():
            return type('Config', (), {
                'mt': type('MTConfig', (), {
                    'wait_k': type('WaitKConfig', (), {
                        'k': 3,
                        'adaptive_k': True,
                        'active': 'dev',
                        'dev_k': 3,
                        'prod_k': 5
                    })()
                })()
            })()


@dataclass
class WaitKState:
    """State information for Wait-k policy"""
    current_k: int  # Current k value
    source_tokens: List[str] = field(default_factory=list)  # Accumulated source tokens
    target_tokens: List[str] = field(default_factory=list)  # Generated target tokens
    read_position: int = 0  # Current read position in source
    write_position: int = 0  # Current write position in target
    is_reading: bool = True  # Whether currently reading or writing
    confidence_history: List[float] = field(default_factory=list)  # Confidence scores
    latency_history: List[float] = field(default_factory=list)  # Latency measurements
    
    def reset(self):
        """Reset the state for new translation"""
        self.source_tokens.clear()
        self.target_tokens.clear()
        self.read_position = 0
        self.write_position = 0
        self.is_reading = True
        self.confidence_history.clear()
        self.latency_history.clear()
    
    def get_lagging_behind(self) -> int:
        """Calculate how far behind target is from source"""
        return len(self.source_tokens) - len(self.target_tokens)
    
    def get_average_confidence(self) -> float:
        """Get average confidence from history"""
        return np.mean(self.confidence_history) if self.confidence_history else 0.0
    
    def get_average_latency(self) -> float:
        """Get average latency from history"""
        return np.mean(self.latency_history) if self.latency_history else 0.0


class WaitKPolicy(ABC):
    """Abstract base class for Wait-k policies"""
    
    def __init__(self, initial_k: int = 3):
        self.initial_k = initial_k
        self.state = WaitKState(current_k=initial_k)
        self.stats = {
            'total_decisions': 0,
            'read_decisions': 0,
            'write_decisions': 0,
            'k_adjustments': 0,
            'average_k': initial_k,
            'total_latency': 0.0
        }
    
    @abstractmethod
    def should_read(self, source_available: bool = True) -> bool:
        """
        Decide whether to read next source token or write target token
        
        Args:
            source_available: Whether more source tokens are available
            
        Returns:
            True if should read, False if should write
        """
        pass
    
    @abstractmethod
    def update_k(self, confidence: Optional[float] = None, latency: Optional[float] = None) -> int:
        """
        Update the k value based on feedback
        
        Args:
            confidence: Translation confidence score
            latency: Current latency measurement
            
        Returns:
            New k value
        """
        pass
    
    def add_source_token(self, token: str) -> None:
        """Add a new source token"""
        self.state.source_tokens.append(token)
        self.state.read_position = len(self.state.source_tokens)
    
    def add_target_token(self, token: str, confidence: Optional[float] = None) -> None:
        """Add a new target token"""
        self.state.target_tokens.append(token)
        self.state.write_position = len(self.state.target_tokens)
        
        if confidence is not None:
            self.state.confidence_history.append(confidence)
    
    def can_start_translation(self) -> bool:
        """Check if we have enough tokens to start translation"""
        return len(self.state.source_tokens) >= self.state.current_k
    
    def get_source_context(self, max_tokens: Optional[int] = None) -> List[str]:
        """Get source tokens available for translation"""
        if max_tokens is None:
            return self.state.source_tokens.copy()
        return self.state.source_tokens[:max_tokens]
    
    def reset(self, new_k: Optional[int] = None) -> None:
        """Reset the policy state"""
        if new_k is not None:
            self.state.current_k = new_k
        self.state.reset()
    
    def get_policy_stats(self) -> Dict:
        """Get policy statistics"""
        total_decisions = self.stats['total_decisions']
        if total_decisions > 0:
            self.stats['read_ratio'] = self.stats['read_decisions'] / total_decisions
            self.stats['write_ratio'] = self.stats['write_decisions'] / total_decisions
        
        return self.stats.copy()


class FixedWaitK(WaitKPolicy):
    """Fixed Wait-k policy - always waits for exactly k tokens"""
    
    def __init__(self, k: int = 3):
        super().__init__(k)
        self.k = k
    
    def should_read(self, source_available: bool = True) -> bool:
        """
        Fixed Wait-k decision: read until we have k tokens, then alternate
        """
        self.stats['total_decisions'] += 1
        
        # If we don't have k tokens yet, always read (if source available)
        if len(self.state.source_tokens) < self.k:
            if source_available:
                self.stats['read_decisions'] += 1
                return True
            else:
                # No more source available, start writing
                self.stats['write_decisions'] += 1
                return False
        
        # After k tokens, use simple alternating strategy
        # Read if we're lagging behind by more than k
        lagging = self.state.get_lagging_behind()
        should_read_decision = lagging > self.k and source_available
        
        if should_read_decision:
            self.stats['read_decisions'] += 1
        else:
            self.stats['write_decisions'] += 1
            
        return should_read_decision
    
    def update_k(self, confidence: Optional[float] = None, latency: Optional[float] = None) -> int:
        """Fixed k doesn't change"""
        if latency is not None:
            self.state.latency_history.append(latency)
        return self.k


class AdaptiveWaitK(WaitKPolicy):
    """Adaptive Wait-k policy - adjusts k based on confidence and latency"""
    
    def __init__(self, initial_k: int = 3, min_k: int = 1, max_k: int = 10,
                 confidence_threshold: float = 0.7, latency_threshold: float = 2.0):
        super().__init__(initial_k)
        self.min_k = min_k
        self.max_k = max_k
        self.confidence_threshold = confidence_threshold
        self.latency_threshold = latency_threshold
        
        # Adaptation parameters
        self.adaptation_window = 10  # Number of decisions to consider
        self.k_increase_factor = 1  # How much to increase k
        self.k_decrease_factor = 1  # How much to decrease k
    
    def should_read(self, source_available: bool = True) -> bool:
        """
        Adaptive decision based on current k and context
        """
        self.stats['total_decisions'] += 1
        
        # If we don't have enough tokens yet, always read (if available)
        if len(self.state.source_tokens) < self.state.current_k:
            if source_available:
                self.stats['read_decisions'] += 1
                return True
            else:
                # No more source, start writing
                self.stats['write_decisions'] += 1
                return False
        
        # Adaptive decision based on quality and latency
        recent_confidence = self._get_recent_confidence()
        recent_latency = self._get_recent_latency()
        lagging = self.state.get_lagging_behind()
        
        # Decision factors
        confidence_pressure = recent_confidence < self.confidence_threshold if recent_confidence else False
        latency_pressure = recent_latency > self.latency_threshold if recent_latency else False
        lagging_pressure = lagging > self.state.current_k
        
        # Read more if quality is low but not if latency is too high
        should_read_decision = (
            source_available and 
            (confidence_pressure and not latency_pressure) or
            (lagging_pressure and not latency_pressure)
        )
        
        if should_read_decision:
            self.stats['read_decisions'] += 1
        else:
            self.stats['write_decisions'] += 1
            
        return should_read_decision
    
    def update_k(self, confidence: Optional[float] = None, latency: Optional[float] = None) -> int:
        """
        Adapt k based on confidence and latency feedback
        """
        old_k = self.state.current_k
        
        if confidence is not None:
            self.state.confidence_history.append(confidence)
        if latency is not None:
            self.state.latency_history.append(latency)
        
        # Only adapt after we have enough history
        if len(self.state.confidence_history) < self.adaptation_window:
            return self.state.current_k
        
        recent_confidence = self._get_recent_confidence()
        recent_latency = self._get_recent_latency()
        
        # Adaptation logic
        if recent_confidence < self.confidence_threshold:
            # Low confidence -> increase k (wait for more context)
            self.state.current_k = min(self.state.current_k + self.k_increase_factor, self.max_k)
        elif recent_latency > self.latency_threshold:
            # High latency -> decrease k (reduce waiting)
            self.state.current_k = max(self.state.current_k - self.k_decrease_factor, self.min_k)
        elif recent_confidence > self.confidence_threshold and recent_latency < self.latency_threshold:
            # Good performance -> can try decreasing k slightly
            if np.random.random() < 0.1:  # 10% chance to explore lower k
                self.state.current_k = max(self.state.current_k - 1, self.min_k)
        
        # Track k changes
        if self.state.current_k != old_k:
            self.stats['k_adjustments'] += 1
        
        # Update average k
        total_decisions = self.stats['total_decisions']
        if total_decisions > 0:
            self.stats['average_k'] = (
                self.stats['average_k'] * (total_decisions - 1) + self.state.current_k
            ) / total_decisions
        
        return self.state.current_k
    
    def _get_recent_confidence(self) -> Optional[float]:
        """Get average confidence over recent window"""
        if not self.state.confidence_history:
            return None
        recent = self.state.confidence_history[-self.adaptation_window:]
        return np.mean(recent)
    
    def _get_recent_latency(self) -> Optional[float]:
        """Get average latency over recent window"""
        if not self.state.latency_history:
            return None
        recent = self.state.latency_history[-self.adaptation_window:]
        return np.mean(recent)


class WaitKSimulator:
    """Simulator for testing Wait-k policies"""
    
    def __init__(self, policy: WaitKPolicy):
        self.policy = policy
        self.simulation_stats = {
            'total_steps': 0,
            'translation_steps': 0,
            'reading_steps': 0,
            'final_latency': 0.0,
            'final_quality': 0.0
        }
    
    def simulate_translation(self, source_tokens: List[str], 
                           target_tokens: List[str],
                           quality_scores: Optional[List[float]] = None) -> Dict:
        """
        Simulate translation process with given tokens
        
        Args:
            source_tokens: List of source tokens
            target_tokens: Expected target tokens  
            quality_scores: Quality score for each target token
            
        Returns:
            Simulation results
        """
        self.policy.reset()
        simulation_log = []
        
        source_idx = 0
        target_idx = 0
        step = 0
        
        while source_idx < len(source_tokens) or target_idx < len(target_tokens):
            step += 1
            self.simulation_stats['total_steps'] = step
            
            # Check if more source is available
            source_available = source_idx < len(source_tokens)
            target_available = target_idx < len(target_tokens)
            
            # Get policy decision
            should_read = self.policy.should_read(source_available)
            
            step_info = {
                'step': step,
                'action': 'read' if should_read else 'write',
                'source_position': source_idx,
                'target_position': target_idx,
                'k_value': self.policy.state.current_k
            }
            
            if should_read and source_available:
                # Read source token
                token = source_tokens[source_idx]
                self.policy.add_source_token(token)
                source_idx += 1
                step_info['token'] = token
                step_info['token_type'] = 'source'
                self.simulation_stats['reading_steps'] += 1
                
            elif target_available:
                # Write target token
                token = target_tokens[target_idx]
                confidence = quality_scores[target_idx] if quality_scores else 0.8
                self.policy.add_target_token(token, confidence)
                
                # Update policy with feedback
                latency = step * 0.1  # Simulated latency
                self.policy.update_k(confidence, latency)
                
                target_idx += 1
                step_info['token'] = token
                step_info['token_type'] = 'target'
                step_info['confidence'] = confidence
                step_info['latency'] = latency
                self.simulation_stats['translation_steps'] += 1
                
            else:
                # Nothing to do
                break
            
            simulation_log.append(step_info)
        
        # Calculate final metrics
        self.simulation_stats['final_latency'] = step * 0.1
        self.simulation_stats['final_quality'] = self.policy.state.get_average_confidence()
        
        return {
            'simulation_log': simulation_log,
            'simulation_stats': self.simulation_stats,
            'policy_stats': self.policy.get_policy_stats(),
            'final_state': {
                'source_tokens': len(self.policy.state.source_tokens),
                'target_tokens': len(self.policy.state.target_tokens),
                'final_k': self.policy.state.current_k
            }
        }


def create_wait_k_policy(policy_type: str = "adaptive", **kwargs) -> WaitKPolicy:
    """
    Factory function to create Wait-k policy
    
    Args:
        policy_type: Type of policy ('fixed' or 'adaptive')
        **kwargs: Policy-specific parameters
        
    Returns:
        WaitKPolicy instance
    """
    try:
        config = get_config().mt.wait_k
        
        if policy_type == "fixed":
            k = kwargs.get('k', config.k)
            return FixedWaitK(k=k)
        elif policy_type == "adaptive":
            initial_k = kwargs.get('initial_k', config.k)
            min_k = kwargs.get('min_k', 1)
            max_k = kwargs.get('max_k', 10)
            return AdaptiveWaitK(
                initial_k=initial_k,
                min_k=min_k,
                max_k=max_k,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
            
    except:
        # Fallback
        if policy_type == "fixed":
            return FixedWaitK(k=kwargs.get('k', 3))
        else:
            return AdaptiveWaitK(initial_k=kwargs.get('initial_k', 3))


def analyze_wait_k_performance(results: Dict) -> Dict:
    """Analyze Wait-k policy performance from simulation results"""
    
    analysis = {}
    
    # Basic metrics
    sim_stats = results['simulation_stats']
    policy_stats = results['policy_stats']
    
    analysis['efficiency'] = {
        'total_steps': sim_stats['total_steps'],
        'reading_ratio': sim_stats['reading_steps'] / sim_stats['total_steps'],
        'translation_ratio': sim_stats['translation_steps'] / sim_stats['total_steps']
    }
    
    analysis['quality'] = {
        'final_confidence': sim_stats['final_quality'],
        'average_k': policy_stats.get('average_k', 0),
        'k_adjustments': policy_stats.get('k_adjustments', 0)
    }
    
    analysis['latency'] = {
        'final_latency': sim_stats['final_latency'],
        'steps_per_target_token': sim_stats['total_steps'] / max(sim_stats['translation_steps'], 1)
    }
    
    # Decision quality
    total_decisions = policy_stats.get('total_decisions', 1)
    analysis['decisions'] = {
        'total_decisions': total_decisions,
        'read_ratio': policy_stats.get('read_ratio', 0),
        'write_ratio': policy_stats.get('write_ratio', 0)
    }
    
    # Overall score (balance of quality and latency)
    quality_score = sim_stats['final_quality']
    latency_penalty = sim_stats['final_latency'] / 10.0  # Normalize latency
    analysis['overall_score'] = max(0, quality_score - latency_penalty)
    
    return analysis


# Utility functions
def compare_wait_k_policies(source_tokens: List[str], 
                          target_tokens: List[str],
                          k_values: List[int] = [1, 3, 5]) -> Dict:
    """Compare different Wait-k policies on the same input"""
    
    results = {}
    
    for k in k_values:
        # Test fixed policy
        policy = FixedWaitK(k=k)
        simulator = WaitKSimulator(policy)
        result = simulator.simulate_translation(source_tokens, target_tokens)
        results[f'fixed_k_{k}'] = analyze_wait_k_performance(result)
    
    # Test adaptive policy
    adaptive_policy = AdaptiveWaitK(initial_k=3)
    adaptive_simulator = WaitKSimulator(adaptive_policy)
    adaptive_result = adaptive_simulator.simulate_translation(source_tokens, target_tokens)
    results['adaptive'] = analyze_wait_k_performance(adaptive_result)
    
    return results 