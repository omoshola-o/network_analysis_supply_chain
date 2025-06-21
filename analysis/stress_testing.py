"""
Network Analysis for Systemic Risk Assessment in Supply Chains
Stress Testing and Simulation Module

This module implements comprehensive stress testing scenarios including:
- Monte Carlo simulations of random and targeted failures
- Liquidity crisis propagation models
- Cross-sector contagion scenarios
- Correlated shock models
- Percolation analysis for cascading failures

Author: Generated Analysis Framework
Date: 2025-06-20
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
from scipy import stats
from scipy.sparse import csr_matrix
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShockType(Enum):
    """Enumeration of different shock types for stress testing."""
    LIQUIDITY_CRISIS = "liquidity_crisis"
    DEMAND_SHOCK = "demand_shock"
    SUPPLY_DISRUPTION = "supply_disruption"
    FINANCIAL_CONTAGION = "financial_contagion"
    OPERATIONAL_FAILURE = "operational_failure"
    REGULATORY_SHOCK = "regulatory_shock"

@dataclass
class StressTestScenario:
    """Container for stress test scenario parameters."""
    scenario_id: str
    shock_type: ShockType
    initial_shocked_nodes: List[str]
    shock_magnitude: float
    propagation_probability: float
    recovery_rate: float
    max_simulation_rounds: int
    scenario_description: str

@dataclass
class StressTestResults:
    """Container for stress test simulation results."""
    scenario_id: str
    initial_failures: List[str]
    final_failed_nodes: List[str]
    cascade_sequence: List[Dict[str, Any]]
    total_failure_rate: float
    economic_impact: float
    recovery_time: int
    network_fragmentation: Dict[str, Any]
    verification_passed: bool
    computation_time: float

class SupplyChainStressTester:
    """
    Comprehensive stress testing framework for supply chain networks.
    Implements various shock scenarios and propagation models.
    """
    
    def __init__(self, network: nx.DiGraph, risk_metrics: Dict[str, Any], 
                 random_seed: int = 42):
        """
        Initialize stress tester with network and risk metrics.
        
        Args:
            network: Supply chain network graph
            risk_metrics: Pre-computed risk metrics for nodes
            random_seed: Seed for reproducible simulations
        """
        self.network = network.copy()
        self.risk_metrics = risk_metrics
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Pre-compute network statistics for efficiency
        self.total_network_value = sum(
            self.network.nodes[node].get('revenue_millions', 0) 
            for node in self.network.nodes()
        )
        
        logger.info(f"Initialized stress tester for network with {self.network.number_of_nodes()} nodes")
    
    def run_monte_carlo_failures(self, n_simulations: int = 1000, 
                                failure_rates: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations of random node failures.
        
        Args:
            n_simulations: Number of simulation runs
            failure_rates: Custom failure rates by node tier
            
        Returns:
            Dictionary with aggregated simulation results
        """
        logger.info(f"Running {n_simulations} Monte Carlo failure simulations")
        
        if failure_rates is None:
            failure_rates = {'S': 0.05, 'M': 0.03, 'R': 0.02}  # Default rates by tier
        
        simulation_results = []
        
        for sim_id in range(n_simulations):
            # Generate random failures based on tier-specific rates
            failed_nodes = []
            for node in self.network.nodes():
                tier = self.network.nodes[node].get('tier', 'M')
                failure_rate = failure_rates.get(tier, 0.03)
                
                # Adjust failure rate based on financial fragility
                if hasattr(self.risk_metrics.get(node, {}), 'financial_fragility'):
                    fragility = self.risk_metrics[node].financial_fragility
                    adjusted_rate = failure_rate * (1 + fragility)
                else:
                    adjusted_rate = failure_rate
                
                if np.random.random() < adjusted_rate:
                    failed_nodes.append(node)
            
            # Simulate cascade from random failures
            cascade_result = self.simulate_cascade_propagation(
                initial_failed_nodes=failed_nodes,
                shock_type=ShockType.OPERATIONAL_FAILURE,
                propagation_threshold=0.3
            )
            
            simulation_results.append({
                'simulation_id': sim_id,
                'initial_failures': len(failed_nodes),
                'final_failures': len(cascade_result['final_failed_nodes']),
                'cascade_length': cascade_result['cascade_length'],
                'economic_impact': cascade_result['economic_impact'],
                'failure_rate': cascade_result['total_failure_rate']
            })
        
        # Aggregate results
        aggregated_results = self._aggregate_monte_carlo_results(simulation_results)
        
        logger.info(f"Monte Carlo simulation completed: {aggregated_results['summary']['mean_failure_rate']:.3f} average failure rate")
        return aggregated_results
    
    def run_targeted_attack_simulation(self, attack_strategies: List[str] = None) -> Dict[str, Any]:
        """
        Simulate targeted attacks on high-centrality nodes.
        
        Args:
            attack_strategies: List of attack strategies to test
            
        Returns:
            Dictionary with attack simulation results
        """
        if attack_strategies is None:
            attack_strategies = ['high_degree', 'high_betweenness', 'high_systemic_importance', 'random']
        
        logger.info(f"Running targeted attack simulations with {len(attack_strategies)} strategies")
        
        attack_results = {}
        
        for strategy in attack_strategies:
            # Select target nodes based on strategy
            target_nodes = self._select_attack_targets(strategy, num_targets=10)
            
            # Simulate progressive attacks
            progressive_results = []
            cumulative_targets = []
            
            for i, target in enumerate(target_nodes):
                cumulative_targets.append(target)
                
                cascade_result = self.simulate_cascade_propagation(
                    initial_failed_nodes=cumulative_targets.copy(),
                    shock_type=ShockType.OPERATIONAL_FAILURE,
                    propagation_threshold=0.25
                )
                
                progressive_results.append({
                    'attack_round': i + 1,
                    'targets_so_far': len(cumulative_targets),
                    'total_failures': len(cascade_result['final_failed_nodes']),
                    'failure_rate': cascade_result['total_failure_rate'],
                    'economic_impact': cascade_result['economic_impact'],
                    'largest_component_size': self._calculate_largest_component_size(
                        cascade_result['final_failed_nodes']
                    )
                })
            
            attack_results[strategy] = {
                'target_nodes': target_nodes,
                'progressive_results': progressive_results,
                'final_impact': progressive_results[-1] if progressive_results else {}
            }
        
        return attack_results
    
    def simulate_liquidity_crisis(self, crisis_severity: float = 0.3, 
                                affected_sectors: List[str] = None) -> Dict[str, Any]:
        """
        Simulate liquidity crisis propagation through the network.
        
        Args:
            crisis_severity: Severity of the liquidity shock (0-1)
            affected_sectors: List of initially affected sectors/tiers
            
        Returns:
            Dictionary with liquidity crisis simulation results
        """
        logger.info(f"Simulating liquidity crisis with severity {crisis_severity}")
        
        if affected_sectors is None:
            affected_sectors = ['S']  # Start with suppliers
        
        # Identify initially affected nodes
        initially_affected = []
        for node in self.network.nodes():
            tier = self.network.nodes[node].get('tier', 'M')
            if tier in affected_sectors:
                # Probability of being affected based on financial fragility
                if hasattr(self.risk_metrics.get(node, {}), 'financial_fragility'):
                    fragility = self.risk_metrics[node].financial_fragility
                    liquidity_ratio = self.network.nodes[node].get('liquidity_ratio', 1.0)
                    
                    # Higher fragility and lower liquidity = higher crisis probability
                    crisis_probability = crisis_severity * fragility * (2 - liquidity_ratio)
                    if np.random.random() < crisis_probability:
                        initially_affected.append(node)
        
        # Simulate liquidity crisis propagation
        crisis_results = self._simulate_liquidity_propagation(
            initially_affected, crisis_severity
        )
        
        return crisis_results
    
    def simulate_correlated_shocks(self, correlation_matrix: np.ndarray = None,
                                 shock_scenarios: List[StressTestScenario] = None) -> Dict[str, Any]:
        """
        Simulate correlated shocks across multiple sectors/regions.
        
        Args:
            correlation_matrix: Correlation matrix for shock propagation
            shock_scenarios: List of correlated shock scenarios
            
        Returns:
            Dictionary with correlated shock simulation results
        """
        logger.info("Simulating correlated shocks across sectors")
        
        if shock_scenarios is None:
            shock_scenarios = self._create_default_shock_scenarios()
        
        if correlation_matrix is None:
            correlation_matrix = self._estimate_sector_correlations()
        
        # Generate correlated random shocks
        n_scenarios = len(shock_scenarios)
        correlated_shocks = np.random.multivariate_normal(
            mean=np.zeros(n_scenarios),
            cov=correlation_matrix,
            size=100  # Number of simulation runs
        )
        
        simulation_results = []
        
        for sim_idx, shock_vector in enumerate(correlated_shocks):
            combined_failed_nodes = set()
            scenario_impacts = {}
            
            # Apply each correlated shock
            for scenario_idx, (scenario, shock_intensity) in enumerate(zip(shock_scenarios, shock_vector)):
                # Convert shock intensity to probability
                shock_probability = stats.norm.cdf(shock_intensity)
                
                # Select affected nodes based on scenario and shock intensity
                affected_nodes = self._select_scenario_affected_nodes(scenario, shock_probability)
                
                # Simulate cascade for this scenario component
                cascade_result = self.simulate_cascade_propagation(
                    initial_failed_nodes=affected_nodes,
                    shock_type=scenario.shock_type,
                    propagation_threshold=scenario.propagation_probability
                )
                
                combined_failed_nodes.update(cascade_result['final_failed_nodes'])
                scenario_impacts[scenario.scenario_id] = cascade_result
            
            # Calculate overall impact
            total_impact = self._calculate_economic_impact(list(combined_failed_nodes))
            
            simulation_results.append({
                'simulation_id': sim_idx,
                'total_failed_nodes': len(combined_failed_nodes),
                'total_failure_rate': len(combined_failed_nodes) / self.network.number_of_nodes(),
                'economic_impact': total_impact,
                'scenario_breakdown': scenario_impacts
            })
        
        return {
            'simulation_results': simulation_results,
            'summary_statistics': self._summarize_correlated_shock_results(simulation_results),
            'correlation_matrix': correlation_matrix.tolist()
        }
    
    def simulate_cascade_propagation(self, initial_failed_nodes: List[str],
                                   shock_type: ShockType,
                                   propagation_threshold: float = 0.3,
                                   max_rounds: int = 50) -> Dict[str, Any]:
        """
        Simulate cascade propagation from initial failures.
        
        Args:
            initial_failed_nodes: List of initially failed nodes
            shock_type: Type of shock causing the cascade
            propagation_threshold: Threshold for cascade propagation
            max_rounds: Maximum simulation rounds
            
        Returns:
            Dictionary with cascade simulation results
        """
        failed_nodes = set(initial_failed_nodes)
        cascade_sequence = []
        
        for round_num in range(max_rounds):
            round_start_failures = len(failed_nodes)
            new_failures = set()
            
            # Check each non-failed node for cascade failure
            for node in self.network.nodes():
                if node in failed_nodes:
                    continue
                
                failure_probability = self._calculate_cascade_failure_probability(
                    node, failed_nodes, shock_type, propagation_threshold
                )
                
                if np.random.random() < failure_probability:
                    new_failures.add(node)
            
            # Record cascade round
            cascade_sequence.append({
                'round': round_num + 1,
                'new_failures': list(new_failures),
                'total_failures': len(failed_nodes) + len(new_failures),
                'failure_rate': (len(failed_nodes) + len(new_failures)) / self.network.number_of_nodes()
            })
            
            failed_nodes.update(new_failures)
            
            # Stop if no new failures
            if len(new_failures) == 0:
                break
        
        # Calculate final impact metrics
        economic_impact = self._calculate_economic_impact(list(failed_nodes))
        network_fragmentation = self._analyze_network_fragmentation(list(failed_nodes))
        
        return {
            'initial_failed_nodes': initial_failed_nodes,
            'final_failed_nodes': list(failed_nodes),
            'cascade_sequence': cascade_sequence,
            'cascade_length': len(cascade_sequence),
            'total_failure_rate': len(failed_nodes) / self.network.number_of_nodes(),
            'economic_impact': economic_impact,
            'network_fragmentation': network_fragmentation
        }
    
    def run_percolation_analysis(self, removal_fractions: List[float] = None) -> Dict[str, Any]:
        """
        Analyze network percolation properties under node removal.
        
        Args:
            removal_fractions: List of node removal fractions to test
            
        Returns:
            Dictionary with percolation analysis results
        """
        if removal_fractions is None:
            removal_fractions = np.linspace(0.0, 0.8, 17)
        
        logger.info(f"Running percolation analysis with {len(removal_fractions)} removal fractions")
        
        percolation_results = []
        
        for fraction in removal_fractions:
            # Number of nodes to remove
            n_remove = int(fraction * self.network.number_of_nodes())
            
            # Multiple runs for statistical significance
            runs_results = []
            for run in range(50):  # 50 runs per fraction
                # Random node removal
                nodes_to_remove = np.random.choice(
                    list(self.network.nodes()), 
                    size=n_remove, 
                    replace=False
                )
                
                # Create reduced network
                reduced_network = self.network.copy()
                reduced_network.remove_nodes_from(nodes_to_remove)
                
                # Analyze connectivity
                if reduced_network.number_of_nodes() > 0:
                    # For directed graphs, use weakly connected components
                    components = list(nx.weakly_connected_components(reduced_network))
                    largest_component_size = max(len(comp) for comp in components) if components else 0
                    num_components = len(components)
                    
                    runs_results.append({
                        'largest_component_fraction': largest_component_size / self.network.number_of_nodes(),
                        'num_components': num_components,
                        'remaining_nodes': reduced_network.number_of_nodes(),
                        'remaining_edges': reduced_network.number_of_edges()
                    })
                else:
                    runs_results.append({
                        'largest_component_fraction': 0.0,
                        'num_components': 0,
                        'remaining_nodes': 0,
                        'remaining_edges': 0
                    })
            
            # Aggregate results for this fraction
            avg_largest_component = np.mean([r['largest_component_fraction'] for r in runs_results])
            avg_num_components = np.mean([r['num_components'] for r in runs_results])
            
            percolation_results.append({
                'removal_fraction': fraction,
                'avg_largest_component_fraction': avg_largest_component,
                'avg_num_components': avg_num_components,
                'std_largest_component': np.std([r['largest_component_fraction'] for r in runs_results]),
                'percolation_threshold_reached': avg_largest_component < 0.1
            })
        
        # Identify percolation threshold
        percolation_threshold = None
        for result in percolation_results:
            if result['percolation_threshold_reached']:
                percolation_threshold = result['removal_fraction']
                break
        
        return {
            'percolation_results': percolation_results,
            'percolation_threshold': percolation_threshold,
            'network_resilience_score': 1 - (percolation_threshold or 0.8)
        }
    
    # Private helper methods
    def _aggregate_monte_carlo_results(self, simulation_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate Monte Carlo simulation results."""
        failure_rates = [r['failure_rate'] for r in simulation_results]
        economic_impacts = [r['economic_impact'] for r in simulation_results]
        cascade_lengths = [r['cascade_length'] for r in simulation_results]
        
        return {
            'summary': {
                'num_simulations': len(simulation_results),
                'mean_failure_rate': np.mean(failure_rates),
                'std_failure_rate': np.std(failure_rates),
                'percentile_failure_rates': np.percentile(failure_rates, [5, 25, 50, 75, 95]),
                'mean_economic_impact': np.mean(economic_impacts),
                'mean_cascade_length': np.mean(cascade_lengths),
                'max_observed_failure_rate': np.max(failure_rates)
            },
            'detailed_results': simulation_results
        }
    
    def _select_attack_targets(self, strategy: str, num_targets: int) -> List[str]:
        """Select target nodes based on attack strategy."""
        if strategy == 'random':
            return np.random.choice(list(self.network.nodes()), size=num_targets, replace=False).tolist()
        
        # Calculate metric for strategy
        if strategy == 'high_degree':
            node_scores = {node: self.network.degree(node) for node in self.network.nodes()}
        elif strategy == 'high_betweenness':
            betweenness = nx.betweenness_centrality(self.network)
            node_scores = betweenness
        elif strategy == 'high_systemic_importance':
            node_scores = {}
            for node in self.network.nodes():
                if hasattr(self.risk_metrics.get(node, {}), 'systemic_importance'):
                    node_scores[node] = self.risk_metrics[node].systemic_importance
                else:
                    node_scores[node] = 0.0
        else:
            # Default to degree centrality
            node_scores = {node: self.network.degree(node) for node in self.network.nodes()}
        
        # Select top nodes
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        return [node for node, score in sorted_nodes[:num_targets]]
    
    def _calculate_largest_component_size(self, failed_nodes: List[str]) -> float:
        """Calculate size of largest connected component after failures."""
        remaining_network = self.network.copy()
        remaining_network.remove_nodes_from(failed_nodes)
        
        if remaining_network.number_of_nodes() == 0:
            return 0.0
        
        components = list(nx.weakly_connected_components(remaining_network))
        largest_component_size = max(len(comp) for comp in components) if components else 0
        
        return largest_component_size / self.network.number_of_nodes()
    
    def _simulate_liquidity_propagation(self, initially_affected: List[str], 
                                      crisis_severity: float) -> Dict[str, Any]:
        """Simulate liquidity crisis propagation."""
        affected_nodes = set(initially_affected)
        propagation_rounds = []
        
        for round_num in range(20):  # Max 20 rounds
            new_affected = set()
            
            for node in self.network.nodes():
                if node in affected_nodes:
                    continue
                
                # Calculate liquidity stress from affected suppliers
                liquidity_stress = 0.0
                total_exposure = 0.0
                
                for predecessor in self.network.predecessors(node):
                    if predecessor in affected_nodes:
                        edge_data = self.network.edges[predecessor, node]
                        exposure = edge_data.get('transaction_volume_millions', 0)
                        payment_delay_risk = edge_data.get('payment_terms_days', 30) / 30.0
                        
                        liquidity_stress += exposure * payment_delay_risk
                    
                    total_exposure += self.network.edges[predecessor, node].get('transaction_volume_millions', 0)
                
                # Node's liquidity buffer
                liquidity_ratio = self.network.nodes[node].get('liquidity_ratio', 1.0)
                working_capital = self.network.nodes[node].get('working_capital_days', 30)
                
                # Calculate liquidity crisis probability
                if total_exposure > 0:
                    normalized_stress = liquidity_stress / total_exposure
                    crisis_probability = (normalized_stress * crisis_severity) / (liquidity_ratio * working_capital / 30)
                    
                    if np.random.random() < crisis_probability:
                        new_affected.add(node)
            
            propagation_rounds.append({
                'round': round_num + 1,
                'new_affected': list(new_affected),
                'total_affected': len(affected_nodes) + len(new_affected)
            })
            
            if not new_affected:
                break
            
            affected_nodes.update(new_affected)
        
        # Calculate economic impact
        economic_impact = self._calculate_economic_impact(list(affected_nodes))
        
        return {
            'initially_affected': initially_affected,
            'final_affected_nodes': list(affected_nodes),
            'propagation_rounds': propagation_rounds,
            'total_affected_rate': len(affected_nodes) / self.network.number_of_nodes(),
            'economic_impact': economic_impact,
            'crisis_severity': crisis_severity
        }
    
    def _create_default_shock_scenarios(self) -> List[StressTestScenario]:
        """Create default shock scenarios for correlated analysis."""
        return [
            StressTestScenario(
                scenario_id="supplier_disruption",
                shock_type=ShockType.SUPPLY_DISRUPTION,
                initial_shocked_nodes=[],
                shock_magnitude=0.4,
                propagation_probability=0.3,
                recovery_rate=0.1,
                max_simulation_rounds=30,
                scenario_description="Major supplier disruption"
            ),
            StressTestScenario(
                scenario_id="demand_collapse",
                shock_type=ShockType.DEMAND_SHOCK,
                initial_shocked_nodes=[],
                shock_magnitude=0.3,
                propagation_probability=0.25,
                recovery_rate=0.15,
                max_simulation_rounds=25,
                scenario_description="Demand collapse in retail sector"
            ),
            StressTestScenario(
                scenario_id="financial_contagion",
                shock_type=ShockType.FINANCIAL_CONTAGION,
                initial_shocked_nodes=[],
                shock_magnitude=0.5,
                propagation_probability=0.4,
                recovery_rate=0.05,
                max_simulation_rounds=40,
                scenario_description="Financial contagion spread"
            )
        ]
    
    def _estimate_sector_correlations(self) -> np.ndarray:
        """Estimate correlation matrix between sectors."""
        # Simple correlation matrix - in practice, this would be estimated from data
        n_scenarios = 3
        correlation_matrix = np.eye(n_scenarios)
        
        # Add some correlation between scenarios
        correlation_matrix[0, 1] = 0.3  # Supply disruption -> Demand shock
        correlation_matrix[1, 0] = 0.3
        correlation_matrix[0, 2] = 0.4  # Supply disruption -> Financial contagion
        correlation_matrix[2, 0] = 0.4
        correlation_matrix[1, 2] = 0.2  # Demand shock -> Financial contagion
        correlation_matrix[2, 1] = 0.2
        
        return correlation_matrix
    
    def _select_scenario_affected_nodes(self, scenario: StressTestScenario, 
                                      shock_intensity: float) -> List[str]:
        """Select nodes affected by a specific scenario."""
        affected_nodes = []
        
        for node in self.network.nodes():
            # Base probability based on shock type and node characteristics
            base_prob = 0.0
            
            if scenario.shock_type == ShockType.SUPPLY_DISRUPTION:
                if self.network.nodes[node].get('tier') == 'S':
                    base_prob = 0.1
            elif scenario.shock_type == ShockType.DEMAND_SHOCK:
                if self.network.nodes[node].get('tier') == 'R':
                    base_prob = 0.08
            elif scenario.shock_type == ShockType.FINANCIAL_CONTAGION:
                if hasattr(self.risk_metrics.get(node, {}), 'financial_fragility'):
                    fragility = self.risk_metrics[node].financial_fragility
                    base_prob = 0.05 * (1 + fragility)
            
            # Adjust by shock intensity
            adjusted_prob = base_prob * (1 + shock_intensity) * scenario.shock_magnitude
            
            if np.random.random() < adjusted_prob:
                affected_nodes.append(node)
        
        return affected_nodes
    
    def _calculate_cascade_failure_probability(self, node: str, failed_nodes: set,
                                             shock_type: ShockType, threshold: float) -> float:
        """Calculate probability of cascade failure for a node."""
        # Base failure probability from direct dependencies
        dependency_stress = 0.0
        total_dependencies = 0.0
        
        for predecessor in self.network.predecessors(node):
            dependency_strength = self.network.edges[predecessor, node].get('dependency_strength', 0)
            total_dependencies += dependency_strength
            
            if predecessor in failed_nodes:
                dependency_stress += dependency_strength
        
        # Normalize dependency stress
        if total_dependencies > 0:
            normalized_stress = dependency_stress / total_dependencies
        else:
            normalized_stress = 0.0
        
        # Adjust for node's financial fragility
        if hasattr(self.risk_metrics.get(node, {}), 'financial_fragility'):
            fragility = self.risk_metrics[node].financial_fragility
            fragility_multiplier = 1 + fragility
        else:
            fragility_multiplier = 1.0
        
        # Shock type specific adjustments
        shock_multiplier = {
            ShockType.LIQUIDITY_CRISIS: 1.5,
            ShockType.FINANCIAL_CONTAGION: 1.3,
            ShockType.SUPPLY_DISRUPTION: 1.2,
            ShockType.OPERATIONAL_FAILURE: 1.0,
            ShockType.DEMAND_SHOCK: 0.8,
            ShockType.REGULATORY_SHOCK: 0.9
        }.get(shock_type, 1.0)
        
        # Calculate final failure probability
        failure_probability = (normalized_stress * fragility_multiplier * shock_multiplier) - threshold
        
        return max(0.0, min(1.0, failure_probability))
    
    def _calculate_economic_impact(self, failed_nodes: List[str]) -> float:
        """Calculate economic impact of failed nodes."""
        total_impact = sum(
            self.network.nodes[node].get('revenue_millions', 0)
            for node in failed_nodes
        )
        
        return total_impact / self.total_network_value if self.total_network_value > 0 else 0.0
    
    def _analyze_network_fragmentation(self, failed_nodes: List[str]) -> Dict[str, Any]:
        """Analyze network fragmentation after failures."""
        remaining_network = self.network.copy()
        remaining_network.remove_nodes_from(failed_nodes)
        
        if remaining_network.number_of_nodes() == 0:
            return {
                'largest_component_size': 0,
                'num_components': 0,
                'fragmentation_index': 1.0,
                'connectivity_loss': 1.0
            }
        
        components = list(nx.weakly_connected_components(remaining_network))
        largest_component_size = max(len(comp) for comp in components) if components else 0
        
        return {
            'largest_component_size': largest_component_size,
            'num_components': len(components),
            'fragmentation_index': 1 - (largest_component_size / self.network.number_of_nodes()),
            'connectivity_loss': 1 - (remaining_network.number_of_edges() / self.network.number_of_edges())
        }
    
    def _summarize_correlated_shock_results(self, simulation_results: List[Dict]) -> Dict[str, Any]:
        """Summarize correlated shock simulation results."""
        failure_rates = [r['total_failure_rate'] for r in simulation_results]
        economic_impacts = [r['economic_impact'] for r in simulation_results]
        
        return {
            'mean_failure_rate': np.mean(failure_rates),
            'std_failure_rate': np.std(failure_rates),
            'percentile_failure_rates': np.percentile(failure_rates, [5, 25, 50, 75, 95]).tolist(),
            'mean_economic_impact': np.mean(economic_impacts),
            'max_failure_rate': np.max(failure_rates),
            'correlation_amplification': np.std(failure_rates) / np.mean(failure_rates) if np.mean(failure_rates) > 0 else 0
        }

def save_stress_test_results(results: Dict[str, Any], output_path: str = "journal_package") -> Dict[str, str]:
    """
    Save stress test results in multiple formats.
    
    Args:
        results: Dictionary containing all stress test results
        output_path: Base output directory
        
    Returns:
        Dictionary with paths to saved files
    """
    file_paths = {}
    
    # Save comprehensive results as JSON
    results_json_path = f"verification_outputs/stress_test_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    file_paths['stress_results_json'] = results_json_path
    
    # Extract key metrics for summary
    summary_metrics = {
        'monte_carlo_summary': results.get('monte_carlo', {}).get('summary', {}),
        'attack_simulation_summary': {
            strategy: result.get('final_impact', {})
            for strategy, result in results.get('targeted_attacks', {}).items()
        },
        'liquidity_crisis_impact': results.get('liquidity_crisis', {}).get('total_affected_rate', 0),
        'percolation_threshold': results.get('percolation_analysis', {}).get('percolation_threshold', None),
        'network_resilience_score': results.get('percolation_analysis', {}).get('network_resilience_score', 0)
    }
    
    # Save summary for LaTeX generation
    summary_json_path = f"stress_test_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary_metrics, f, indent=2, default=str)
    file_paths['summary_json'] = summary_json_path
    
    logger.info(f"Stress test results saved to: {list(file_paths.values())}")
    return file_paths

if __name__ == "__main__":
    # Example usage and testing
    from data_preprocessing import SupplyChainDataGenerator
    from statistical_analysis import SystemicRiskAnalyzer
    
    # Generate test network
    generator = SupplyChainDataGenerator(seed=42)
    test_network = generator.generate_synthetic_network(n_suppliers=100, n_manufacturers=30, n_retailers=50)
    
    # Compute risk metrics
    risk_analyzer = SystemicRiskAnalyzer(test_network)
    risk_metrics = risk_analyzer.compute_comprehensive_risk_metrics()
    
    # Initialize stress tester
    stress_tester = SupplyChainStressTester(test_network, risk_metrics, random_seed=42)
    
    # Run comprehensive stress tests
    all_results = {}
    
    # Monte Carlo simulations
    all_results['monte_carlo'] = stress_tester.run_monte_carlo_failures(n_simulations=500)
    
    # Targeted attacks
    all_results['targeted_attacks'] = stress_tester.run_targeted_attack_simulation()
    
    # Liquidity crisis
    all_results['liquidity_crisis'] = stress_tester.simulate_liquidity_crisis(crisis_severity=0.4)
    
    # Percolation analysis
    all_results['percolation_analysis'] = stress_tester.run_percolation_analysis()
    
    # Save results
    file_paths = save_stress_test_results(all_results)
    
    print(f"Stress testing completed. Results saved to: {list(file_paths.values())}")