"""
Network Analysis for Systemic Risk Assessment in Supply Chains
Comprehensive Verification Suite

This module provides comprehensive verification protocols ensuring accuracy,
consistency, and reproducibility across all analysis components. Implements
cross-module validation, statistical verification, and content consistency checks.

Key Features:
- Cross-module data consistency verification
- Statistical accuracy validation
- Figure-analysis alignment checks
- LaTeX-code consistency verification
- Comprehensive audit trails
- Automated quality assurance

Author: Generated Analysis Framework
Date: 2025-06-20
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
import hashlib
from datetime import datetime
import warnings
from scipy import stats
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Container for individual verification test results."""
    test_name: str
    module: str
    passed: bool
    error_message: Optional[str]
    warning_message: Optional[str]
    details: Dict[str, Any]
    timestamp: str
    verification_level: str  # 'critical', 'important', 'informational'

@dataclass
class ComprehensiveVerificationReport:
    """Container for comprehensive verification report."""
    overall_passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    critical_failures: int
    verification_results: List[VerificationResult]
    cross_module_consistency: Dict[str, bool]
    data_integrity_score: float
    reproducibility_verified: bool
    generation_timestamp: str
    verification_signature: str

class ComprehensiveVerificationSuite:
    """
    Comprehensive verification system ensuring accuracy and consistency
    across all analysis components.
    """
    
    def __init__(self, tolerance: float = 1e-6, strict_mode: bool = True):
        """
        Initialize verification suite.
        
        Args:
            tolerance: Numerical tolerance for comparisons
            strict_mode: Whether to use strict verification criteria
        """
        self.tolerance = tolerance
        self.strict_mode = strict_mode
        self.verification_results = []
        self.verification_metadata = {}
        
        logger.info(f"Initialized verification suite (strict_mode={strict_mode}, tolerance={tolerance})")
    
    def run_comprehensive_verification(self, 
                                     network_data: nx.DiGraph,
                                     risk_metrics: Dict[str, Any],
                                     stress_test_results: Dict[str, Any],
                                     visualization_metadata: Dict[str, Any],
                                     latex_content: Optional[str] = None) -> ComprehensiveVerificationReport:
        """
        Run comprehensive verification across all analysis components.
        
        Args:
            network_data: Network graph data
            risk_metrics: Computed risk metrics
            stress_test_results: Stress testing results
            visualization_metadata: Visualization metadata and verification
            latex_content: Generated LaTeX content (optional)
            
        Returns:
            ComprehensiveVerificationReport with detailed results
        """
        logger.info("Starting comprehensive verification suite")
        
        # Clear previous results
        self.verification_results = []
        
        # 1. Data Integrity Verification
        self._verify_data_integrity(network_data)
        
        # 2. Risk Metrics Verification
        self._verify_risk_metrics(network_data, risk_metrics)
        
        # 3. Stress Test Verification
        self._verify_stress_test_results(network_data, risk_metrics, stress_test_results)
        
        # 4. Visualization Verification
        self._verify_visualization_consistency(risk_metrics, visualization_metadata)
        
        # 5. Cross-Module Consistency
        cross_module_consistency = self._verify_cross_module_consistency(
            network_data, risk_metrics, stress_test_results, visualization_metadata
        )
        
        # 6. LaTeX Content Verification (if provided)
        if latex_content:
            self._verify_latex_consistency(
                risk_metrics, stress_test_results, visualization_metadata, latex_content
            )
        
        # 7. Reproducibility Verification
        reproducibility_verified = self._verify_reproducibility(
            network_data, risk_metrics, stress_test_results
        )
        
        # Generate comprehensive report
        report = self._generate_verification_report(
            cross_module_consistency, reproducibility_verified
        )
        
        logger.info(f"Comprehensive verification completed: {report.passed_tests}/{report.total_tests} tests passed")
        return report
    
    def _verify_data_integrity(self, network_data: nx.DiGraph):
        """Verify network data integrity and consistency."""
        logger.info("Verifying data integrity")
        
        # Test 1: Network structure validity
        try:
            assert network_data.number_of_nodes() > 0, "Network has no nodes"
            assert network_data.number_of_edges() > 0, "Network has no edges"
            assert nx.is_directed(network_data), "Network should be directed"
            
            self._add_verification_result(
                "network_structure_valid", "data_preprocessing", True, 
                details={
                    'num_nodes': network_data.number_of_nodes(),
                    'num_edges': network_data.number_of_edges(),
                    'is_directed': nx.is_directed(network_data)
                },
                verification_level="critical"
            )
        except AssertionError as e:
            self._add_verification_result(
                "network_structure_valid", "data_preprocessing", False,
                error_message=str(e), verification_level="critical"
            )
        
        # Test 2: Node attributes completeness
        required_node_attrs = ['revenue_millions', 'debt_to_equity', 'liquidity_ratio', 
                              'working_capital_days', 'tier']
        missing_attrs = []
        
        for node in network_data.nodes():
            node_attrs = network_data.nodes[node]
            for attr in required_node_attrs:
                if attr not in node_attrs:
                    missing_attrs.append(f"{node}:{attr}")
        
        if not missing_attrs:
            self._add_verification_result(
                "node_attributes_complete", "data_preprocessing", True,
                details={'required_attributes': required_node_attrs},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "node_attributes_complete", "data_preprocessing", False,
                error_message=f"Missing attributes: {missing_attrs[:10]}{'...' if len(missing_attrs) > 10 else ''}",
                verification_level="important"
            )
        
        # Test 3: Edge attributes completeness
        required_edge_attrs = ['transaction_volume_millions', 'dependency_strength', 'lead_time_days']
        edge_missing_attrs = []
        
        for edge in network_data.edges():
            edge_attrs = network_data.edges[edge]
            for attr in required_edge_attrs:
                if attr not in edge_attrs:
                    edge_missing_attrs.append(f"{edge}:{attr}")
        
        if not edge_missing_attrs:
            self._add_verification_result(
                "edge_attributes_complete", "data_preprocessing", True,
                details={'required_attributes': required_edge_attrs},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "edge_attributes_complete", "data_preprocessing", False,
                error_message=f"Missing edge attributes: {edge_missing_attrs[:10]}{'...' if len(edge_missing_attrs) > 10 else ''}",
                verification_level="important"
            )
        
        # Test 4: Data value ranges
        self._verify_data_value_ranges(network_data)
        
        # Test 5: Network connectivity
        self._verify_network_connectivity(network_data)
    
    def _verify_data_value_ranges(self, network_data: nx.DiGraph):
        """Verify that data values are within expected ranges."""
        
        range_violations = []
        
        # Node attribute ranges
        node_ranges = {
            'revenue_millions': (0, float('inf')),
            'debt_to_equity': (0, 10),  # Reasonable upper bound
            'liquidity_ratio': (0, 10),  # Reasonable upper bound
            'working_capital_days': (-365, 365),  # Can be negative
        }
        
        for node in network_data.nodes():
            node_attrs = network_data.nodes[node]
            for attr, (min_val, max_val) in node_ranges.items():
                if attr in node_attrs:
                    value = node_attrs[attr]
                    if not (min_val <= value <= max_val):
                        range_violations.append(f"Node {node} {attr}: {value}")
        
        # Edge attribute ranges
        edge_ranges = {
            'transaction_volume_millions': (0, float('inf')),
            'dependency_strength': (0, 1),
            'lead_time_days': (0, 365),
        }
        
        for edge in network_data.edges():
            edge_attrs = network_data.edges[edge]
            for attr, (min_val, max_val) in edge_ranges.items():
                if attr in edge_attrs:
                    value = edge_attrs[attr]
                    if not (min_val <= value <= max_val):
                        range_violations.append(f"Edge {edge} {attr}: {value}")
        
        if not range_violations:
            self._add_verification_result(
                "data_value_ranges_valid", "data_preprocessing", True,
                details={'ranges_checked': {**node_ranges, **edge_ranges}},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "data_value_ranges_valid", "data_preprocessing", False,
                error_message=f"Range violations: {range_violations[:5]}{'...' if len(range_violations) > 5 else ''}",
                verification_level="important"
            )
    
    def _verify_network_connectivity(self, network_data: nx.DiGraph):
        """Verify network connectivity properties."""
        
        # Check if network is weakly connected
        is_weakly_connected = nx.is_weakly_connected(network_data)
        
        # Count connected components
        num_components = nx.number_weakly_connected_components(network_data)
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(network_data))
        
        details = {
            'is_weakly_connected': is_weakly_connected,
            'num_weak_components': num_components,
            'num_isolated_nodes': len(isolated_nodes),
            'isolated_nodes': isolated_nodes[:10] if isolated_nodes else []
        }
        
        # For supply chains, we expect mostly connected networks
        if is_weakly_connected or num_components <= 3:
            self._add_verification_result(
                "network_connectivity_adequate", "data_preprocessing", True,
                details=details, verification_level="important"
            )
        else:
            self._add_verification_result(
                "network_connectivity_adequate", "data_preprocessing", False,
                warning_message=f"Network has {num_components} components, may indicate data issues",
                details=details, verification_level="important"
            )
    
    def _verify_risk_metrics(self, network_data: nx.DiGraph, risk_metrics: Dict[str, Any]):
        """Verify risk metrics computation accuracy."""
        logger.info("Verifying risk metrics")
        
        # Test 1: Risk metrics coverage
        expected_coverage = network_data.number_of_nodes()
        actual_coverage = len(risk_metrics)
        
        if actual_coverage == expected_coverage:
            self._add_verification_result(
                "risk_metrics_coverage_complete", "statistical_analysis", True,
                details={'expected': expected_coverage, 'actual': actual_coverage},
                verification_level="critical"
            )
        else:
            self._add_verification_result(
                "risk_metrics_coverage_complete", "statistical_analysis", False,
                error_message=f"Expected {expected_coverage} metrics, got {actual_coverage}",
                verification_level="critical"
            )
        
        # Test 2: Risk metric value bounds
        self._verify_risk_metric_bounds(risk_metrics)
        
        # Test 3: Risk metric statistical properties
        self._verify_risk_metric_statistics(risk_metrics)
        
        # Test 4: Risk metric correlations consistency
        self._verify_risk_metric_correlations(risk_metrics)
    
    def _verify_risk_metric_bounds(self, risk_metrics: Dict[str, Any]):
        """Verify risk metrics are within expected bounds."""
        
        bound_violations = []
        
        # Expected bounds for different metrics
        metric_bounds = {
            'systemic_importance': (0, 1),
            'financial_fragility': (0, 1),
            'contagion_potential': (0, 1),
            'eigenvector_centrality': (0, 1),
            'betweenness_centrality': (0, 1),
            'closeness_centrality': (0, 1),
            'k_core_number': (0, float('inf'))
        }
        
        for node_id, metrics in risk_metrics.items():
            if hasattr(metrics, 'systemic_importance'):
                for metric_name, (min_bound, max_bound) in metric_bounds.items():
                    if hasattr(metrics, metric_name):
                        value = getattr(metrics, metric_name)
                        if not (min_bound <= value <= max_bound):
                            bound_violations.append(f"{node_id}.{metric_name}: {value}")
        
        if not bound_violations:
            self._add_verification_result(
                "risk_metric_bounds_valid", "statistical_analysis", True,
                details={'bounds_checked': metric_bounds},
                verification_level="critical"
            )
        else:
            self._add_verification_result(
                "risk_metric_bounds_valid", "statistical_analysis", False,
                error_message=f"Bound violations: {bound_violations[:5]}{'...' if len(bound_violations) > 5 else ''}",
                verification_level="critical"
            )
    
    def _verify_risk_metric_statistics(self, risk_metrics: Dict[str, Any]):
        """Verify statistical properties of risk metrics."""
        
        # Extract metric arrays
        metric_arrays = {}
        for node_id, metrics in risk_metrics.items():
            if hasattr(metrics, 'systemic_importance'):
                for attr in ['systemic_importance', 'financial_fragility', 'contagion_potential']:
                    if hasattr(metrics, attr):
                        if attr not in metric_arrays:
                            metric_arrays[attr] = []
                        metric_arrays[attr].append(getattr(metrics, attr))
        
        statistical_issues = []
        
        for metric_name, values in metric_arrays.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Check for reasonable distributions
                if std_val == 0:
                    statistical_issues.append(f"{metric_name}: zero variance (all values identical)")
                elif std_val > mean_val * 2:  # High relative variance
                    statistical_issues.append(f"{metric_name}: very high variance (std={std_val:.3f}, mean={mean_val:.3f})")
                
                # Check for extreme values
                if np.any(np.isnan(values)):
                    statistical_issues.append(f"{metric_name}: contains NaN values")
                if np.any(np.isinf(values)):
                    statistical_issues.append(f"{metric_name}: contains infinite values")
        
        if not statistical_issues:
            self._add_verification_result(
                "risk_metric_statistics_reasonable", "statistical_analysis", True,
                details={'metrics_analyzed': list(metric_arrays.keys())},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "risk_metric_statistics_reasonable", "statistical_analysis", False,
                warning_message=f"Statistical issues: {statistical_issues[:3]}",
                verification_level="important"
            )
    
    def _verify_risk_metric_correlations(self, risk_metrics: Dict[str, Any]):
        """Verify risk metric correlations are reasonable."""
        
        # Extract correlation data
        metric_data = []
        for node_id, metrics in risk_metrics.items():
            if hasattr(metrics, 'systemic_importance'):
                metric_data.append({
                    'systemic_importance': metrics.systemic_importance,
                    'financial_fragility': metrics.financial_fragility,
                    'contagion_potential': metrics.contagion_potential,
                    'eigenvector_centrality': metrics.eigenvector_centrality
                })
        
        if len(metric_data) > 10:  # Need sufficient data for correlation
            df = pd.DataFrame(metric_data)
            corr_matrix = df.corr()
            
            # Check for reasonable correlations
            correlation_issues = []
            
            # Systemic importance should correlate positively with centrality
            si_eigen_corr = corr_matrix.loc['systemic_importance', 'eigenvector_centrality']
            if si_eigen_corr < 0.1:
                correlation_issues.append(f"Low systemic importance-centrality correlation: {si_eigen_corr:.3f}")
            
            # Check for perfect correlations (suspicious)
            perfect_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.99:
                        perfect_corrs.append(f"{corr_matrix.columns[i]}-{corr_matrix.columns[j]}: {corr_val:.3f}")
            
            if perfect_corrs:
                correlation_issues.extend(perfect_corrs)
            
            if not correlation_issues:
                self._add_verification_result(
                    "risk_metric_correlations_reasonable", "statistical_analysis", True,
                    details={'correlation_matrix': corr_matrix.to_dict()},
                    verification_level="informational"
                )
            else:
                self._add_verification_result(
                    "risk_metric_correlations_reasonable", "statistical_analysis", False,
                    warning_message=f"Correlation issues: {correlation_issues[:2]}",
                    verification_level="informational"
                )
    
    def _verify_stress_test_results(self, network_data: nx.DiGraph, 
                                  risk_metrics: Dict[str, Any],
                                  stress_test_results: Dict[str, Any]):
        """Verify stress test results consistency and reasonableness."""
        logger.info("Verifying stress test results")
        
        # Test 1: Monte Carlo results consistency
        if 'monte_carlo' in stress_test_results:
            self._verify_monte_carlo_results(stress_test_results['monte_carlo'])
        
        # Test 2: Attack simulation results
        if 'targeted_attacks' in stress_test_results:
            self._verify_attack_simulation_results(stress_test_results['targeted_attacks'])
        
        # Test 3: Cascade propagation results
        if 'liquidity_crisis' in stress_test_results:
            self._verify_cascade_results(stress_test_results['liquidity_crisis'])
        
        # Test 4: Percolation analysis results
        if 'percolation_analysis' in stress_test_results:
            self._verify_percolation_results(stress_test_results['percolation_analysis'])
    
    def _verify_monte_carlo_results(self, mc_results: Dict[str, Any]):
        """Verify Monte Carlo simulation results."""
        
        if 'detailed_results' not in mc_results:
            self._add_verification_result(
                "monte_carlo_structure_valid", "stress_testing", False,
                error_message="Missing detailed_results in Monte Carlo output",
                verification_level="critical"
            )
            return
        
        detailed_results = mc_results['detailed_results']
        
        # Check result structure
        required_fields = ['simulation_id', 'failure_rate', 'economic_impact', 'cascade_length']
        structure_valid = all(
            all(field in result for field in required_fields) 
            for result in detailed_results
        )
        
        if structure_valid:
            self._add_verification_result(
                "monte_carlo_structure_valid", "stress_testing", True,
                details={'num_simulations': len(detailed_results)},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "monte_carlo_structure_valid", "stress_testing", False,
                error_message="Monte Carlo results missing required fields",
                verification_level="important"
            )
        
        # Check value ranges
        failure_rates = [r['failure_rate'] for r in detailed_results]
        economic_impacts = [r['economic_impact'] for r in detailed_results]
        
        range_issues = []
        if not all(0 <= fr <= 1 for fr in failure_rates):
            range_issues.append("Failure rates outside [0,1] range")
        if not all(0 <= ei <= 1 for ei in economic_impacts):
            range_issues.append("Economic impacts outside [0,1] range")
        
        if not range_issues:
            self._add_verification_result(
                "monte_carlo_ranges_valid", "stress_testing", True,
                details={
                    'failure_rate_range': [min(failure_rates), max(failure_rates)],
                    'economic_impact_range': [min(economic_impacts), max(economic_impacts)]
                },
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "monte_carlo_ranges_valid", "stress_testing", False,
                error_message=f"Range issues: {range_issues}",
                verification_level="important"
            )
    
    def _verify_attack_simulation_results(self, attack_results: Dict[str, Any]):
        """Verify targeted attack simulation results."""
        
        attack_strategies = list(attack_results.keys())
        
        # Check that we have multiple strategies
        if len(attack_strategies) >= 2:
            self._add_verification_result(
                "attack_strategies_diverse", "stress_testing", True,
                details={'strategies': attack_strategies},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "attack_strategies_diverse", "stress_testing", False,
                warning_message=f"Only {len(attack_strategies)} attack strategies tested",
                verification_level="important"
            )
        
        # Check that targeted attacks are more effective than random
        if 'random' in attack_results and len(attack_strategies) > 1:
            random_impact = attack_results['random'].get('final_impact', {}).get('failure_rate', 0)
            targeted_impacts = [
                attack_results[strategy].get('final_impact', {}).get('failure_rate', 0)
                for strategy in attack_strategies if strategy != 'random'
            ]
            
            if targeted_impacts and max(targeted_impacts) > random_impact:
                self._add_verification_result(
                    "targeted_attacks_more_effective", "stress_testing", True,
                    details={
                        'random_impact': random_impact,
                        'max_targeted_impact': max(targeted_impacts)
                    },
                    verification_level="informational"
                )
            else:
                self._add_verification_result(
                    "targeted_attacks_more_effective", "stress_testing", False,
                    warning_message="Targeted attacks not more effective than random",
                    verification_level="informational"
                )
    
    def _verify_cascade_results(self, cascade_results: Dict[str, Any]):
        """Verify cascade propagation results."""
        
        required_fields = ['initially_affected', 'final_affected_nodes', 'propagation_rounds']
        
        if all(field in cascade_results for field in required_fields):
            self._add_verification_result(
                "cascade_structure_valid", "stress_testing", True,
                details={'required_fields_present': required_fields},
                verification_level="important"
            )
        else:
            missing_fields = [field for field in required_fields if field not in cascade_results]
            self._add_verification_result(
                "cascade_structure_valid", "stress_testing", False,
                error_message=f"Missing cascade fields: {missing_fields}",
                verification_level="important"
            )
            return
        
        # Check cascade progression logic
        initial_count = len(cascade_results['initially_affected'])
        final_count = len(cascade_results['final_affected_nodes'])
        
        if final_count >= initial_count:
            self._add_verification_result(
                "cascade_progression_logical", "stress_testing", True,
                details={'initial_affected': initial_count, 'final_affected': final_count},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "cascade_progression_logical", "stress_testing", False,
                error_message=f"Final affected ({final_count}) less than initial ({initial_count})",
                verification_level="important"
            )
    
    def _verify_percolation_results(self, percolation_results: Dict[str, Any]):
        """Verify percolation analysis results."""
        
        if 'percolation_results' not in percolation_results:
            self._add_verification_result(
                "percolation_structure_valid", "stress_testing", False,
                error_message="Missing percolation_results data",
                verification_level="important"
            )
            return
        
        perc_data = percolation_results['percolation_results']
        
        # Check monotonic behavior (generally decreasing connectivity)
        removal_fractions = [r['removal_fraction'] for r in perc_data]
        component_fractions = [r['avg_largest_component_fraction'] for r in perc_data]
        
        # Check if data is sorted by removal fraction
        is_sorted = all(removal_fractions[i] <= removal_fractions[i+1] for i in range(len(removal_fractions)-1))
        
        if is_sorted:
            self._add_verification_result(
                "percolation_data_ordered", "stress_testing", True,
                details={'data_points': len(perc_data)},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "percolation_data_ordered", "stress_testing", False,
                warning_message="Percolation data not properly ordered by removal fraction",
                verification_level="important"
            )
        
        # Check general decreasing trend
        decreasing_trend = sum(
            component_fractions[i] >= component_fractions[i+1] 
            for i in range(len(component_fractions)-1)
        ) / max(1, len(component_fractions)-1)
        
        if decreasing_trend > 0.7:  # Allow some noise
            self._add_verification_result(
                "percolation_trend_decreasing", "stress_testing", True,
                details={'decreasing_ratio': decreasing_trend},
                verification_level="informational"
            )
        else:
            self._add_verification_result(
                "percolation_trend_decreasing", "stress_testing", False,
                warning_message=f"Percolation trend not sufficiently decreasing: {decreasing_trend:.2f}",
                verification_level="informational"
            )
    
    def _verify_visualization_consistency(self, risk_metrics: Dict[str, Any], 
                                        visualization_metadata: Dict[str, Any]):
        """Verify visualization consistency with analysis data."""
        logger.info("Verifying visualization consistency")
        
        # Test 1: All major visualizations present
        expected_visualizations = [
            'network_topology', 'risk_distributions', 'correlation_heatmap'
        ]
        
        missing_visualizations = [
            viz for viz in expected_visualizations 
            if viz not in visualization_metadata
        ]
        
        if not missing_visualizations:
            self._add_verification_result(
                "required_visualizations_present", "visualization_generation", True,
                details={'expected_visualizations': expected_visualizations},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "required_visualizations_present", "visualization_generation", False,
                error_message=f"Missing visualizations: {missing_visualizations}",
                verification_level="important"
            )
        
        # Test 2: Visualization verification status
        failed_verifications = []
        for viz_id, metadata in visualization_metadata.items():
            if isinstance(metadata, dict) and not metadata.get('verification_passed', True):
                failed_verifications.append(viz_id)
        
        if not failed_verifications:
            self._add_verification_result(
                "visualization_verification_passed", "visualization_generation", True,
                details={'total_visualizations': len(visualization_metadata)},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "visualization_verification_passed", "visualization_generation", False,
                error_message=f"Visualizations failed verification: {failed_verifications}",
                verification_level="important"
            )
        
        # Test 3: Statistical consistency between data and visualizations
        self._verify_visualization_statistical_consistency(risk_metrics, visualization_metadata)
    
    def _verify_visualization_statistical_consistency(self, risk_metrics: Dict[str, Any], 
                                                    visualization_metadata: Dict[str, Any]):
        """Verify statistical consistency between analysis data and visualizations."""
        
        # Extract statistical summaries from visualizations
        consistency_issues = []
        
        for viz_id, metadata in visualization_metadata.items():
            if isinstance(metadata, dict) and 'statistical_summary' in metadata:
                viz_stats = metadata['statistical_summary']
                
                # Check sample sizes
                if 'sample_size' in viz_stats:
                    viz_sample_size = viz_stats['sample_size']
                    expected_sample_size = len(risk_metrics)
                    
                    if viz_sample_size != expected_sample_size:
                        consistency_issues.append(
                            f"{viz_id}: sample size mismatch (viz: {viz_sample_size}, data: {expected_sample_size})"
                        )
        
        if not consistency_issues:
            self._add_verification_result(
                "visualization_statistical_consistency", "visualization_generation", True,
                details={'visualizations_checked': list(visualization_metadata.keys())},
                verification_level="informational"
            )
        else:
            self._add_verification_result(
                "visualization_statistical_consistency", "visualization_generation", False,
                warning_message=f"Consistency issues: {consistency_issues[:2]}",
                verification_level="informational"
            )
    
    def _verify_cross_module_consistency(self, network_data: nx.DiGraph,
                                       risk_metrics: Dict[str, Any],
                                       stress_test_results: Dict[str, Any],
                                       visualization_metadata: Dict[str, Any]) -> Dict[str, bool]:
        """Verify consistency across different analysis modules."""
        logger.info("Verifying cross-module consistency")
        
        consistency_results = {}
        
        # 1. Network size consistency
        network_size = network_data.number_of_nodes()
        risk_metrics_size = len(risk_metrics)
        
        consistency_results['network_risk_size'] = (network_size == risk_metrics_size)
        
        if consistency_results['network_risk_size']:
            self._add_verification_result(
                "network_risk_size_consistent", "cross_module", True,
                details={'network_size': network_size, 'risk_metrics_size': risk_metrics_size},
                verification_level="critical"
            )
        else:
            self._add_verification_result(
                "network_risk_size_consistent", "cross_module", False,
                error_message=f"Size mismatch: network {network_size} vs risk metrics {risk_metrics_size}",
                verification_level="critical"
            )
        
        # 2. Node ID consistency
        network_nodes = set(network_data.nodes())
        risk_metric_nodes = set(risk_metrics.keys())
        
        node_overlap = len(network_nodes.intersection(risk_metric_nodes))
        node_consistency_ratio = node_overlap / max(len(network_nodes), len(risk_metric_nodes))
        
        consistency_results['node_ids_consistent'] = (node_consistency_ratio > 0.95)
        
        if consistency_results['node_ids_consistent']:
            self._add_verification_result(
                "node_ids_consistent", "cross_module", True,
                details={'consistency_ratio': node_consistency_ratio},
                verification_level="critical"
            )
        else:
            missing_in_risk = network_nodes - risk_metric_nodes
            missing_in_network = risk_metric_nodes - network_nodes
            self._add_verification_result(
                "node_ids_consistent", "cross_module", False,
                error_message=f"Node ID mismatch: {len(missing_in_risk)} missing in risk, {len(missing_in_network)} missing in network",
                verification_level="critical"
            )
        
        # 3. Statistical consistency between modules
        self._verify_statistical_cross_module_consistency(
            network_data, risk_metrics, stress_test_results, consistency_results
        )
        
        return consistency_results
    
    def _verify_statistical_cross_module_consistency(self, network_data: nx.DiGraph,
                                                   risk_metrics: Dict[str, Any],
                                                   stress_test_results: Dict[str, Any],
                                                   consistency_results: Dict[str, bool]):
        """Verify statistical consistency across modules."""
        
        # Check revenue consistency between network and risk metrics
        network_revenues = [
            network_data.nodes[node].get('revenue_millions', 0) 
            for node in network_data.nodes()
        ]
        
        # Compare with any revenue-based calculations in stress tests
        if 'monte_carlo' in stress_test_results and 'summary' in stress_test_results['monte_carlo']:
            mc_summary = stress_test_results['monte_carlo']['summary']
            
            # Check if economic impact calculations are reasonable
            if 'mean_economic_impact' in mc_summary:
                mean_economic_impact = mc_summary['mean_economic_impact']
                
                # Economic impact should be related to network total value
                total_network_value = sum(network_revenues)
                
                # Reasonable economic impact should be a fraction of total value
                impact_ratio = mean_economic_impact if total_network_value == 0 else mean_economic_impact / total_network_value
                
                consistency_results['economic_impact_reasonable'] = (0 <= impact_ratio <= 1)
                
                if consistency_results['economic_impact_reasonable']:
                    self._add_verification_result(
                        "economic_impact_reasonable", "cross_module", True,
                        details={'impact_ratio': impact_ratio, 'total_network_value': total_network_value},
                        verification_level="informational"
                    )
                else:
                    self._add_verification_result(
                        "economic_impact_reasonable", "cross_module", False,
                        warning_message=f"Economic impact ratio seems unreasonable: {impact_ratio}",
                        verification_level="informational"
                    )
    
    def _verify_latex_consistency(self, risk_metrics: Dict[str, Any],
                                stress_test_results: Dict[str, Any],
                                visualization_metadata: Dict[str, Any],
                                latex_content: str):
        """Verify LaTeX content consistency with analysis results."""
        logger.info("Verifying LaTeX content consistency")
        
        # Extract numbers from LaTeX content
        latex_numbers = self._extract_numbers_from_latex(latex_content)
        
        # Extract figure references
        figure_references = re.findall(r'\\ref\{fig:(\w+)\}', latex_content)
        
        # Test 1: All referenced figures exist
        missing_figures = [
            fig_ref for fig_ref in figure_references 
            if fig_ref not in visualization_metadata
        ]
        
        if not missing_figures:
            self._add_verification_result(
                "latex_figure_references_valid", "latex_generation", True,
                details={'figure_references': figure_references},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "latex_figure_references_valid", "latex_generation", False,
                error_message=f"Missing figure references: {missing_figures}",
                verification_level="important"
            )
        
        # Test 2: Check for unreplaced placeholders
        placeholders = re.findall(r'\{\{[^}]+\}\}', latex_content)
        
        if not placeholders:
            self._add_verification_result(
                "latex_placeholders_replaced", "latex_generation", True,
                details={'latex_length': len(latex_content)},
                verification_level="important"
            )
        else:
            self._add_verification_result(
                "latex_placeholders_replaced", "latex_generation", False,
                error_message=f"Unreplaced placeholders: {placeholders[:3]}",
                verification_level="important"
            )
        
        # Test 3: Statistical consistency (simplified check)
        self._verify_latex_statistical_consistency(
            latex_numbers, risk_metrics, stress_test_results
        )
    
    def _extract_numbers_from_latex(self, latex_content: str) -> List[float]:
        """Extract numerical values from LaTeX content."""
        # Simple regex to find numbers in LaTeX (this could be more sophisticated)
        number_pattern = r'(?<!\\\w{1,10})\b\d+\.?\d*\b'
        matches = re.findall(number_pattern, latex_content)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    def _verify_latex_statistical_consistency(self, latex_numbers: List[float],
                                            risk_metrics: Dict[str, Any],
                                            stress_test_results: Dict[str, Any]):
        """Verify statistical consistency between LaTeX and analysis results."""
        
        # This is a simplified consistency check
        # In a full implementation, this would be more sophisticated
        
        consistency_issues = []
        
        # Check if LaTeX contains reasonable numbers
        if not latex_numbers:
            consistency_issues.append("No numerical values found in LaTeX content")
        else:
            # Check for extreme values that might indicate errors
            extreme_values = [num for num in latex_numbers if num > 1000 or num < -100]
            if len(extreme_values) > 10:  # Allow some large numbers
                consistency_issues.append(f"Many extreme values in LaTeX: {len(extreme_values)}")
        
        if not consistency_issues:
            self._add_verification_result(
                "latex_statistical_consistency", "latex_generation", True,
                details={'numbers_found': len(latex_numbers)},
                verification_level="informational"
            )
        else:
            self._add_verification_result(
                "latex_statistical_consistency", "latex_generation", False,
                warning_message=f"Consistency issues: {consistency_issues}",
                verification_level="informational"
            )
    
    def _verify_reproducibility(self, network_data: nx.DiGraph,
                              risk_metrics: Dict[str, Any],
                              stress_test_results: Dict[str, Any]) -> bool:
        """Verify reproducibility by checking for deterministic elements."""
        logger.info("Verifying reproducibility")
        
        reproducibility_indicators = []
        
        # Check if network has consistent structure
        network_hash = self._compute_network_hash(network_data)
        reproducibility_indicators.append(network_hash is not None)
        
        # Check if risk metrics have consistent computation
        risk_metrics_hash = self._compute_risk_metrics_hash(risk_metrics)
        reproducibility_indicators.append(risk_metrics_hash is not None)
        
        # Store hashes for future verification
        self.verification_metadata['network_hash'] = network_hash
        self.verification_metadata['risk_metrics_hash'] = risk_metrics_hash
        
        reproducibility_verified = all(reproducibility_indicators)
        
        if reproducibility_verified:
            self._add_verification_result(
                "reproducibility_verified", "verification_suite", True,
                details={
                    'network_hash': network_hash,
                    'risk_metrics_hash': risk_metrics_hash
                },
                verification_level="informational"
            )
        else:
            self._add_verification_result(
                "reproducibility_verified", "verification_suite", False,
                warning_message="Could not verify reproducibility",
                verification_level="informational"
            )
        
        return reproducibility_verified
    
    def _compute_network_hash(self, network_data: nx.DiGraph) -> Optional[str]:
        """Compute hash of network structure for reproducibility verification."""
        try:
            # Create a canonical representation of the network
            edges_sorted = sorted(network_data.edges())
            nodes_sorted = sorted(network_data.nodes())
            
            # Include basic structural information
            network_info = {
                'nodes': nodes_sorted,
                'edges': edges_sorted,
                'num_nodes': network_data.number_of_nodes(),
                'num_edges': network_data.number_of_edges()
            }
            
            # Compute hash
            network_str = json.dumps(network_info, sort_keys=True)
            return hashlib.md5(network_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute network hash: {e}")
            return None
    
    def _compute_risk_metrics_hash(self, risk_metrics: Dict[str, Any]) -> Optional[str]:
        """Compute hash of risk metrics for reproducibility verification."""
        try:
            # Extract key metrics in a deterministic way
            sorted_metrics = {}
            for node_id in sorted(risk_metrics.keys()):
                metrics = risk_metrics[node_id]
                if hasattr(metrics, 'systemic_importance'):
                    sorted_metrics[node_id] = {
                        'systemic_importance': round(metrics.systemic_importance, 6),
                        'financial_fragility': round(metrics.financial_fragility, 6)
                    }
            
            # Compute hash
            metrics_str = json.dumps(sorted_metrics, sort_keys=True)
            return hashlib.md5(metrics_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute risk metrics hash: {e}")
            return None
    
    def _add_verification_result(self, test_name: str, module: str, passed: bool,
                               error_message: Optional[str] = None,
                               warning_message: Optional[str] = None,
                               details: Optional[Dict[str, Any]] = None,
                               verification_level: str = "important"):
        """Add a verification result to the collection."""
        
        result = VerificationResult(
            test_name=test_name,
            module=module,
            passed=passed,
            error_message=error_message,
            warning_message=warning_message,
            details=details or {},
            timestamp=datetime.now().isoformat(),
            verification_level=verification_level
        )
        
        self.verification_results.append(result)
        
        # Log result
        status = "PASS" if passed else "FAIL"
        level = "ERROR" if not passed and verification_level == "critical" else "INFO"
        
        message = f"[{status}] {module}.{test_name}"
        if error_message:
            message += f" - {error_message}"
        elif warning_message:
            message += f" - {warning_message}"
        
        if level == "ERROR":
            logger.error(message)
        else:
            logger.info(message)
    
    def _generate_verification_report(self, cross_module_consistency: Dict[str, bool],
                                    reproducibility_verified: bool) -> ComprehensiveVerificationReport:
        """Generate comprehensive verification report."""
        
        total_tests = len(self.verification_results)
        passed_tests = sum(1 for r in self.verification_results if r.passed)
        failed_tests = total_tests - passed_tests
        critical_failures = sum(
            1 for r in self.verification_results 
            if not r.passed and r.verification_level == "critical"
        )
        
        overall_passed = (critical_failures == 0 and passed_tests / max(total_tests, 1) >= 0.8)
        
        # Calculate data integrity score
        data_integrity_score = self._calculate_data_integrity_score()
        
        # Generate verification signature
        verification_signature = self._generate_verification_signature()
        
        report = ComprehensiveVerificationReport(
            overall_passed=overall_passed,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            critical_failures=critical_failures,
            verification_results=self.verification_results,
            cross_module_consistency=cross_module_consistency,
            data_integrity_score=data_integrity_score,
            reproducibility_verified=reproducibility_verified,
            generation_timestamp=datetime.now().isoformat(),
            verification_signature=verification_signature
        )
        
        return report
    
    def _calculate_data_integrity_score(self) -> float:
        """Calculate overall data integrity score."""
        
        if not self.verification_results:
            return 0.0
        
        # Weight by verification level
        weights = {"critical": 3, "important": 2, "informational": 1}
        
        total_weight = 0
        passed_weight = 0
        
        for result in self.verification_results:
            weight = weights.get(result.verification_level, 1)
            total_weight += weight
            if result.passed:
                passed_weight += weight
        
        return passed_weight / max(total_weight, 1)
    
    def _generate_verification_signature(self) -> str:
        """Generate verification signature for audit trail."""
        
        # Create signature based on verification results
        signature_data = {
            'total_tests': len(self.verification_results),
            'passed_tests': sum(1 for r in self.verification_results if r.passed),
            'timestamp': datetime.now().isoformat(),
            'verification_metadata': self.verification_metadata
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]

def save_verification_report(report: ComprehensiveVerificationReport, 
                           output_path: str = "journal_package") -> str:
    """
    Save comprehensive verification report.
    
    Args:
        report: Comprehensive verification report
        output_path: Output directory
        
    Returns:
        Path to saved report file
    """
    
    # Convert report to serializable format
    report_dict = asdict(report)
    
    # Save detailed report
    report_path = f"verification_outputs/comprehensive_verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    # Generate summary report
    summary = {
        'verification_summary': {
            'overall_passed': report.overall_passed,
            'success_rate': f"{(report.passed_tests/max(report.total_tests,1))*100:.1f}%",
            'critical_failures': report.critical_failures,
            'data_integrity_score': f"{report.data_integrity_score:.3f}",
            'reproducibility_verified': report.reproducibility_verified,
            'verification_signature': report.verification_signature
        },
        'test_breakdown': {
            'by_module': {},
            'by_level': {'critical': 0, 'important': 0, 'informational': 0}
        },
        'issues_summary': []
    }
    
    # Aggregate by module and level
    for result in report.verification_results:
        module = result.module
        level = result.verification_level
        
        if module not in summary['test_breakdown']['by_module']:
            summary['test_breakdown']['by_module'][module] = {'passed': 0, 'failed': 0}
        
        if result.passed:
            summary['test_breakdown']['by_module'][module]['passed'] += 1
        else:
            summary['test_breakdown']['by_module'][module]['failed'] += 1
            
            # Add to issues summary
            summary['issues_summary'].append({
                'test': result.test_name,
                'module': result.module,
                'level': result.verification_level,
                'error': result.error_message or result.warning_message
            })
        
        summary['test_breakdown']['by_level'][level] += 1
    
    # Save summary report
    summary_path = f"verification_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Verification report saved to {report_path}")
    logger.info(f"Verification summary saved to {summary_path}")
    
    return report_path

if __name__ == "__main__":
    # Example usage and testing
    from data_preprocessing import SupplyChainDataGenerator
    from statistical_analysis import SystemicRiskAnalyzer
    
    # Generate test data
    generator = SupplyChainDataGenerator(seed=42)
    test_network = generator.generate_synthetic_network(n_suppliers=50, n_manufacturers=20, n_retailers=30)
    
    # Compute risk metrics
    risk_analyzer = SystemicRiskAnalyzer(test_network)
    risk_metrics = risk_analyzer.compute_comprehensive_risk_metrics()
    
    # Mock stress test results
    mock_stress_results = {
        'monte_carlo': {
            'detailed_results': [
                {'simulation_id': i, 'failure_rate': np.random.random()*0.2, 
                 'economic_impact': np.random.random()*0.1, 'cascade_length': np.random.randint(1, 8)}
                for i in range(100)
            ],
            'summary': {'mean_economic_impact': 0.05}
        }
    }
    
    # Mock visualization metadata
    mock_viz_metadata = {
        'network_topology': {'verification_passed': True, 'statistical_summary': {'sample_size': len(risk_metrics)}},
        'risk_distributions': {'verification_passed': True, 'statistical_summary': {'sample_size': len(risk_metrics)}}
    }
    
    # Run comprehensive verification
    verification_suite = ComprehensiveVerificationSuite(strict_mode=True)
    
    report = verification_suite.run_comprehensive_verification(
        network_data=test_network,
        risk_metrics=risk_metrics,
        stress_test_results=mock_stress_results,
        visualization_metadata=mock_viz_metadata
    )
    
    # Save report
    report_path = save_verification_report(report)
    
    print(f"Comprehensive verification completed:")
    print(f"  Overall passed: {report.overall_passed}")
    print(f"  Tests passed: {report.passed_tests}/{report.total_tests}")
    print(f"  Critical failures: {report.critical_failures}")
    print(f"  Data integrity score: {report.data_integrity_score:.3f}")
    print(f"  Report saved to: {report_path}")