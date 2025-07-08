"""
Network Analysis for Systemic Risk Assessment in Supply Chains
Statistical Analysis Module

This is where I implement all the core statistical methods for the research.
I adapted financial systemic risk models like DebtRank to work with supply
chain networks, which was the key innovation of my approach.

Key algorithms I implemented:
- DebtRank adaptation for supply chains
- Network centrality calculations
- Financial contagion models
- Cross-sector spillover analysis
- Mathematical validation checks

Author: Omoshola S. Owolabi
Date: 2024-12-21
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import warnings
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Container for computed risk metrics with verification metadata."""
    node_id: str
    systemic_importance: float
    debt_rank_score: float
    eigenvector_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    k_core_number: int
    financial_fragility: float
    contagion_potential: float
    verification_passed: bool
    computation_timestamp: str

class SystemicRiskAnalyzer:
    """
    Comprehensive systemic risk analysis for supply chain networks.
    Implements financial contagion models adapted for supply chain operations.
    """
    
    def __init__(self, network: nx.DiGraph, verification_mode: str = "strict"):
        """
        Initialize analyzer with network and verification settings.
        
        Args:
            network: Supply chain network graph
            verification_mode: "strict", "moderate", or "basic" verification level
        """
        self.network = network.copy()
        self.verification_mode = verification_mode
        self.risk_metrics = {}
        self.verification_log = []
        
        # Precompute adjacency matrix for efficiency
        self.adjacency_matrix = nx.adjacency_matrix(self.network, weight='dependency_strength')
        
        logger.info(f"Initialized SystemicRiskAnalyzer with {self.network.number_of_nodes()} nodes")
    
    def compute_debt_rank_scores(self, shock_magnitude: float = 0.1) -> Dict[str, float]:
        """
        Compute DebtRank-style systemic importance scores adapted for supply chains.
        
        DebtRank measures the fraction of total economic value in the network
        that could be impacted by the distress of each node.
        
        Args:
            shock_magnitude: Initial shock size (0-1)
            
        Returns:
            Dictionary mapping node IDs to DebtRank scores
        """
        logger.info("Computing DebtRank systemic importance scores")
        
        debt_rank_scores = {}
        total_network_value = sum(self.network.nodes[node].get('revenue_millions', 0) 
                                for node in self.network.nodes())
        
        for initial_node in self.network.nodes():
            # Initialize impact vector
            impact = {node: 0.0 for node in self.network.nodes()}
            impact[initial_node] = shock_magnitude
            
            # Propagate impact through iterations
            for iteration in range(100):  # Max iterations to prevent infinite loops
                old_impact = impact.copy()
                
                for node in self.network.nodes():
                    if impact[node] > 0:
                        # Calculate impact on downstream nodes
                        for successor in self.network.successors(node):
                            dependency = self.network.edges[node, successor].get('dependency_strength', 0)
                            financial_exposure = self._calculate_financial_exposure(node, successor)
                            
                            additional_impact = impact[node] * dependency * financial_exposure
                            impact[successor] = min(1.0, impact[successor] + additional_impact)
                
                # Check convergence
                if self._impact_converged(old_impact, impact, tolerance=1e-6):
                    break
            
            # Calculate DebtRank as weighted impact
            debt_rank = sum(impact[node] * self.network.nodes[node].get('revenue_millions', 0) 
                          for node in self.network.nodes()) / total_network_value
            
            debt_rank_scores[initial_node] = debt_rank
        
        # Verification
        if self.verification_mode in ["strict", "moderate"]:
            self._verify_debt_rank_scores(debt_rank_scores)
        
        return debt_rank_scores
    
    def compute_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """
        Compute multiple centrality measures for network analysis.
        
        Returns:
            Dictionary with centrality measures for each node
        """
        logger.info("Computing network centrality measures")
        
        centrality_measures = {}
        
        try:
            # Eigenvector centrality (weighted by dependency strength)
            eigenvector_cent = nx.eigenvector_centrality(
                self.network, weight='dependency_strength', max_iter=1000
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge, using unweighted version")
            eigenvector_cent = nx.eigenvector_centrality(self.network, max_iter=1000)
        
        # Betweenness centrality
        betweenness_cent = nx.betweenness_centrality(self.network, weight='dependency_strength')
        
        # Closeness centrality
        closeness_cent = nx.closeness_centrality(self.network, distance='lead_time_days')
        
        # K-core decomposition
        k_core = nx.core_number(self.network.to_undirected())
        
        # PageRank (supply chain adapted)
        pagerank = nx.pagerank(self.network, weight='transaction_volume_millions')
        
        # Combine measures
        for node in self.network.nodes():
            centrality_measures[node] = {
                'eigenvector_centrality': eigenvector_cent.get(node, 0),
                'betweenness_centrality': betweenness_cent.get(node, 0),
                'closeness_centrality': closeness_cent.get(node, 0),
                'k_core_number': k_core.get(node, 0),
                'pagerank': pagerank.get(node, 0)
            }
        
        # Verification
        if self.verification_mode in ["strict", "moderate"]:
            self._verify_centrality_measures(centrality_measures)
        
        return centrality_measures
    
    def compute_financial_fragility_index(self) -> Dict[str, float]:
        """
        Compute financial fragility index combining multiple financial indicators.
        
        Returns:
            Dictionary mapping node IDs to fragility scores (0-1, higher = more fragile)
        """
        logger.info("Computing financial fragility indices")
        
        fragility_scores = {}
        
        for node in self.network.nodes():
            node_attrs = self.network.nodes[node]
            
            # Financial ratio components
            debt_ratio = min(1.0, node_attrs.get('debt_to_equity', 0) / 2.0)  # Normalized to 0-1
            liquidity_stress = max(0.0, 1.0 - node_attrs.get('liquidity_ratio', 1) / 2.0)
            working_capital_pressure = max(0.0, (node_attrs.get('working_capital_days', 30) - 30) / 60)
            
            # Operational vulnerability
            supplier_concentration = 1.0 - min(1.0, node_attrs.get('supplier_diversification', 1) / 10.0)
            customer_concentration = node_attrs.get('customer_concentration', 0.5)
            
            # Size vulnerability (smaller firms more fragile)
            size_penalty = {'Small': 0.3, 'Medium': 0.1, 'Large': 0.0}.get(
                node_attrs.get('company_size', 'Medium'), 0.1
            )
            
            # Weighted fragility score
            fragility = (0.25 * debt_ratio + 
                        0.20 * liquidity_stress + 
                        0.15 * working_capital_pressure +
                        0.20 * supplier_concentration + 
                        0.15 * customer_concentration + 
                        0.05 * size_penalty)
            
            fragility_scores[node] = min(1.0, fragility)
        
        return fragility_scores
    
    def identify_too_central_to_fail(self, threshold_percentile: float = 0.85) -> List[str]:
        """
        Identify 'too-central-to-fail' suppliers based on multiple centrality measures.
        
        Args:
            threshold_percentile: Percentile threshold for classification
            
        Returns:
            List of node IDs classified as too-central-to-fail
        """
        logger.info("Identifying too-central-to-fail suppliers")
        
        centrality_measures = self.compute_centrality_measures()
        debt_rank_scores = self.compute_debt_rank_scores()
        
        # Create composite centrality score
        composite_scores = {}
        for node in self.network.nodes():
            centrality = centrality_measures[node]
            composite_score = (0.3 * centrality['eigenvector_centrality'] +
                             0.25 * centrality['betweenness_centrality'] +
                             0.2 * centrality['pagerank'] +
                             0.25 * debt_rank_scores.get(node, 0))
            composite_scores[node] = composite_score
        
        # Determine threshold
        threshold = np.percentile(list(composite_scores.values()), threshold_percentile * 100)
        
        # Classify nodes
        critical_suppliers = [node for node, score in composite_scores.items() 
                            if score >= threshold]
        
        logger.info(f"Identified {len(critical_suppliers)} too-central-to-fail suppliers")
        return critical_suppliers
    
    def analyze_cross_sector_spillovers(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze potential spillover effects between different sectors/tiers.
        
        Returns:
            Dictionary with spillover matrices between sectors
        """
        logger.info("Analyzing cross-sector spillover effects")
        
        # Group nodes by tier
        tier_groups = {}
        for node in self.network.nodes():
            tier = self.network.nodes[node].get('tier', 'Unknown')
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append(node)
        
        spillover_matrix = {}
        
        for source_tier in tier_groups:
            spillover_matrix[source_tier] = {}
            
            for target_tier in tier_groups:
                if source_tier == target_tier:
                    spillover_matrix[source_tier][target_tier] = 1.0
                    continue
                
                # Calculate average spillover strength
                total_spillover = 0.0
                connection_count = 0
                
                for source_node in tier_groups[source_tier]:
                    for target_node in tier_groups[target_tier]:
                        if self.network.has_edge(source_node, target_node):
                            edge_data = self.network.edges[source_node, target_node]
                            spillover_strength = (edge_data.get('dependency_strength', 0) * 
                                                np.log1p(edge_data.get('transaction_volume_millions', 0)))
                            total_spillover += spillover_strength
                            connection_count += 1
                
                avg_spillover = total_spillover / connection_count if connection_count > 0 else 0.0
                spillover_matrix[source_tier][target_tier] = avg_spillover
        
        return spillover_matrix
    
    def simulate_cascade_failures(self, initial_failed_nodes: List[str], 
                                failure_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Simulate cascading failures starting from initial failed nodes.
        
        Args:
            initial_failed_nodes: List of initially failed node IDs
            failure_threshold: Threshold for cascade propagation
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Simulating cascade failures from {len(initial_failed_nodes)} initial nodes")
        
        failed_nodes = set(initial_failed_nodes)
        cascade_rounds = []
        
        # Get financial fragility scores
        fragility_scores = self.compute_financial_fragility_index()
        
        round_num = 0
        while True:
            round_num += 1
            new_failures = set()
            
            # Check each non-failed node for failure risk
            for node in self.network.nodes():
                if node in failed_nodes:
                    continue
                
                # Calculate stress from failed suppliers
                supplier_stress = 0.0
                total_supplier_dependency = 0.0
                
                for predecessor in self.network.predecessors(node):
                    if predecessor in failed_nodes:
                        dependency = self.network.edges[predecessor, node].get('dependency_strength', 0)
                        supplier_stress += dependency
                    total_supplier_dependency += self.network.edges[predecessor, node].get('dependency_strength', 0)
                
                # Normalize stress
                if total_supplier_dependency > 0:
                    normalized_stress = supplier_stress / total_supplier_dependency
                else:
                    normalized_stress = 0
                
                # Apply failure threshold adjusted by financial fragility
                adjusted_threshold = failure_threshold * (2 - fragility_scores.get(node, 0.5))
                
                if normalized_stress > adjusted_threshold:
                    new_failures.add(node)
            
            cascade_rounds.append({
                'round': round_num,
                'new_failures': list(new_failures),
                'total_failed': len(failed_nodes) + len(new_failures)
            })
            
            if not new_failures:
                break
            
            failed_nodes.update(new_failures)
            
            # Safety check to prevent infinite loops
            if round_num > 50:
                logger.warning("Cascade simulation terminated after 50 rounds")
                break
        
        return {
            'initial_failures': initial_failed_nodes,
            'final_failed_nodes': list(failed_nodes),
            'cascade_rounds': cascade_rounds,
            'total_failure_rate': len(failed_nodes) / self.network.number_of_nodes(),
            'cascade_length': len(cascade_rounds)
        }
    
    def compute_comprehensive_risk_metrics(self) -> Dict[str, RiskMetrics]:
        """
        Compute comprehensive risk metrics for all nodes with verification.
        
        Returns:
            Dictionary mapping node IDs to RiskMetrics objects
        """
        logger.info("Computing comprehensive risk metrics for all nodes")
        
        # Compute individual metric components
        debt_rank_scores = self.compute_debt_rank_scores()
        centrality_measures = self.compute_centrality_measures()
        fragility_scores = self.compute_financial_fragility_index()
        
        comprehensive_metrics = {}
        
        for node in self.network.nodes():
            centrality = centrality_measures[node]
            
            # Calculate contagion potential
            contagion_potential = self._calculate_contagion_potential(node, fragility_scores)
            
            # Verify individual metric
            verification_passed = self._verify_node_metrics(
                node, debt_rank_scores.get(node, 0), centrality, 
                fragility_scores.get(node, 0), contagion_potential
            )
            
            risk_metrics = RiskMetrics(
                node_id=node,
                systemic_importance=debt_rank_scores.get(node, 0),
                debt_rank_score=debt_rank_scores.get(node, 0),
                eigenvector_centrality=centrality['eigenvector_centrality'],
                betweenness_centrality=centrality['betweenness_centrality'],
                closeness_centrality=centrality['closeness_centrality'],
                k_core_number=centrality['k_core_number'],
                financial_fragility=fragility_scores.get(node, 0),
                contagion_potential=contagion_potential,
                verification_passed=verification_passed,
                computation_timestamp=pd.Timestamp.now().isoformat()
            )
            
            comprehensive_metrics[node] = risk_metrics
        
        return comprehensive_metrics
    
    # Private helper methods
    def _calculate_financial_exposure(self, source_node: str, target_node: str) -> float:
        """Calculate financial exposure between two nodes."""
        edge_data = self.network.edges[source_node, target_node]
        transaction_volume = edge_data.get('transaction_volume_millions', 0)
        target_revenue = self.network.nodes[target_node].get('revenue_millions', 1)
        
        # Financial exposure as fraction of target's revenue
        exposure = min(1.0, transaction_volume / max(target_revenue, 0.1))
        return exposure
    
    def _impact_converged(self, old_impact: Dict, new_impact: Dict, tolerance: float) -> bool:
        """Check if impact propagation has converged."""
        max_change = max(abs(new_impact[node] - old_impact[node]) for node in old_impact)
        return max_change < tolerance
    
    def _calculate_contagion_potential(self, node: str, fragility_scores: Dict[str, float]) -> float:
        """Calculate contagion potential for a node."""
        out_degree = self.network.out_degree(node)
        if out_degree == 0:
            return 0.0
        
        # Average fragility of downstream nodes weighted by dependency
        total_weighted_fragility = 0.0
        total_weight = 0.0
        
        for successor in self.network.successors(node):
            dependency = self.network.edges[node, successor].get('dependency_strength', 0)
            fragility = fragility_scores.get(successor, 0.5)
            total_weighted_fragility += dependency * fragility
            total_weight += dependency
        
        if total_weight > 0:
            avg_downstream_fragility = total_weighted_fragility / total_weight
        else:
            avg_downstream_fragility = 0.5
        
        # Contagion potential combines out-degree and downstream fragility
        contagion_potential = np.sqrt(out_degree / 10.0) * avg_downstream_fragility
        return min(1.0, contagion_potential)
    
    # Verification methods
    def _verify_debt_rank_scores(self, debt_rank_scores: Dict[str, float]) -> bool:
        """Verify DebtRank score computation."""
        verification_passed = True
        
        # Check score bounds
        for node, score in debt_rank_scores.items():
            if not (0 <= score <= 1):
                logger.error(f"DebtRank score out of bounds for node {node}: {score}")
                verification_passed = False
        
        # Check for reasonable distribution
        scores = list(debt_rank_scores.values())
        if len(scores) > 0:
            mean_score = np.mean(scores)
            if mean_score > 0.5:
                logger.warning(f"Unusually high average DebtRank score: {mean_score}")
        
        self.verification_log.append({
            'metric': 'debt_rank_scores',
            'passed': verification_passed,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        return verification_passed
    
    def _verify_centrality_measures(self, centrality_measures: Dict[str, Dict[str, float]]) -> bool:
        """Verify centrality measure computation."""
        verification_passed = True
        
        for node, measures in centrality_measures.items():
            # Check bounds for bounded measures
            if not (0 <= measures['eigenvector_centrality'] <= 1):
                logger.error(f"Eigenvector centrality out of bounds for node {node}")
                verification_passed = False
            
            if not (0 <= measures['betweenness_centrality'] <= 1):
                logger.error(f"Betweenness centrality out of bounds for node {node}")
                verification_passed = False
        
        self.verification_log.append({
            'metric': 'centrality_measures',
            'passed': verification_passed,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        return verification_passed
    
    def _verify_node_metrics(self, node: str, debt_rank: float, centrality: Dict, 
                           fragility: float, contagion: float) -> bool:
        """Verify individual node metrics."""
        verification_checks = [
            0 <= debt_rank <= 1,
            0 <= fragility <= 1,
            0 <= contagion <= 1,
            all(0 <= v <= 1 for v in centrality.values() if v is not None)
        ]
        
        return all(verification_checks)

class RiskMetricsValidator:
    """
    Comprehensive validation functions for risk metrics and mathematical computations.
    """
    
    @staticmethod
    def validate_network_metrics(risk_metrics: Dict[str, RiskMetrics]) -> Dict[str, Any]:
        """
        Validate computed risk metrics across the entire network.
        
        Args:
            risk_metrics: Dictionary of computed risk metrics
            
        Returns:
            Validation report with detailed results
        """
        validation_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_nodes_analyzed': len(risk_metrics),
            'validation_results': {},
            'statistical_summary': {},
            'anomaly_detection': {}
        }
        
        # Extract metric arrays for analysis
        systemic_importance = [rm.systemic_importance for rm in risk_metrics.values()]
        financial_fragility = [rm.financial_fragility for rm in risk_metrics.values()]
        contagion_potential = [rm.contagion_potential for rm in risk_metrics.values()]
        eigenvector_centrality = [rm.eigenvector_centrality for rm in risk_metrics.values()]
        
        # Statistical validation
        validation_report['statistical_summary'] = {
            'systemic_importance': {
                'mean': np.mean(systemic_importance),
                'std': np.std(systemic_importance),
                'min': np.min(systemic_importance),
                'max': np.max(systemic_importance),
                'percentiles': np.percentile(systemic_importance, [25, 50, 75, 90, 95])
            },
            'financial_fragility': {
                'mean': np.mean(financial_fragility),
                'std': np.std(financial_fragility),
                'min': np.min(financial_fragility),
                'max': np.max(financial_fragility),
                'percentiles': np.percentile(financial_fragility, [25, 50, 75, 90, 95])
            },
            'contagion_potential': {
                'mean': np.mean(contagion_potential),
                'std': np.std(contagion_potential),
                'min': np.min(contagion_potential),
                'max': np.max(contagion_potential),
                'percentiles': np.percentile(contagion_potential, [25, 50, 75, 90, 95])
            }
        }
        
        # Bounds validation
        validation_report['validation_results']['bounds_check'] = {
            'systemic_importance_valid': all(0 <= x <= 1 for x in systemic_importance),
            'financial_fragility_valid': all(0 <= x <= 1 for x in financial_fragility),
            'contagion_potential_valid': all(0 <= x <= 1 for x in contagion_potential),
            'eigenvector_centrality_valid': all(0 <= x <= 1 for x in eigenvector_centrality)
        }
        
        # Correlation analysis
        validation_report['validation_results']['correlations'] = {
            'fragility_vs_systemic': stats.pearsonr(financial_fragility, systemic_importance)[0],
            'centrality_vs_systemic': stats.pearsonr(eigenvector_centrality, systemic_importance)[0],
            'fragility_vs_contagion': stats.pearsonr(financial_fragility, contagion_potential)[0]
        }
        
        # Anomaly detection using z-scores
        validation_report['anomaly_detection'] = RiskMetricsValidator._detect_metric_anomalies(
            risk_metrics, threshold=3.0
        )
        
        logger.info("Risk metrics validation completed")
        return validation_report
    
    @staticmethod
    def _detect_metric_anomalies(risk_metrics: Dict[str, RiskMetrics], 
                               threshold: float = 3.0) -> Dict[str, List[str]]:
        """Detect anomalous metric values using statistical thresholds."""
        anomalies = {
            'systemic_importance_outliers': [],
            'financial_fragility_outliers': [],
            'contagion_potential_outliers': []
        }
        
        # Extract arrays
        metrics_arrays = {
            'systemic_importance': [rm.systemic_importance for rm in risk_metrics.values()],
            'financial_fragility': [rm.financial_fragility for rm in risk_metrics.values()],
            'contagion_potential': [rm.contagion_potential for rm in risk_metrics.values()]
        }
        
        node_ids = list(risk_metrics.keys())
        
        for metric_name, values in metrics_arrays.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val > 0:
                z_scores = [(v - mean_val) / std_val for v in values]
                outlier_indices = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
                anomalies[f'{metric_name}_outliers'] = [node_ids[i] for i in outlier_indices]
        
        return anomalies

def save_risk_analysis_results(risk_metrics: Dict[str, RiskMetrics], 
                             validation_report: Dict[str, Any],
                             output_path: str = "journal_package") -> Dict[str, str]:
    """
    Save risk analysis results in multiple formats for further analysis and LaTeX generation.
    
    Args:
        risk_metrics: Computed risk metrics
        validation_report: Validation results
        output_path: Base output directory
        
    Returns:
        Dictionary with paths to saved files
    """
    file_paths = {}
    
    # Convert risk metrics to DataFrame
    risk_df_data = []
    for node_id, metrics in risk_metrics.items():
        risk_df_data.append({
            'node_id': metrics.node_id,
            'systemic_importance': metrics.systemic_importance,
            'debt_rank_score': metrics.debt_rank_score,
            'eigenvector_centrality': metrics.eigenvector_centrality,
            'betweenness_centrality': metrics.betweenness_centrality,
            'closeness_centrality': metrics.closeness_centrality,
            'k_core_number': metrics.k_core_number,
            'financial_fragility': metrics.financial_fragility,
            'contagion_potential': metrics.contagion_potential,
            'verification_passed': metrics.verification_passed
        })
    
    risk_df = pd.DataFrame(risk_df_data)
    
    # Save risk metrics CSV
    risk_csv_path = f"risk_metrics.csv"
    risk_df.to_csv(risk_csv_path, index=False)
    file_paths['risk_metrics_csv'] = risk_csv_path
    
    # Save validation report JSON
    validation_json_path = f"verification_outputs/risk_validation.json"
    with open(validation_json_path, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    file_paths['validation_json'] = validation_json_path
    
    # Save summary statistics for LaTeX
    summary_stats = {
        'total_nodes': len(risk_metrics),
        'high_risk_nodes': len([rm for rm in risk_metrics.values() if rm.systemic_importance > 0.1]),
        'critical_nodes': len([rm for rm in risk_metrics.values() if rm.systemic_importance > 0.2]),
        'fragile_nodes': len([rm for rm in risk_metrics.values() if rm.financial_fragility > 0.7]),
        'avg_systemic_importance': np.mean([rm.systemic_importance for rm in risk_metrics.values()]),
        'avg_financial_fragility': np.mean([rm.financial_fragility for rm in risk_metrics.values()]),
        'verification_pass_rate': np.mean([rm.verification_passed for rm in risk_metrics.values()])
    }
    
    summary_json_path = f"summary_statistics.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    file_paths['summary_stats'] = summary_json_path
    
    logger.info(f"Risk analysis results saved to: {list(file_paths.values())}")
    return file_paths

if __name__ == "__main__":
    # Example usage and testing
    from data_preprocessing import SupplyChainDataGenerator
    
    # Generate test network
    generator = SupplyChainDataGenerator(seed=42)
    test_network = generator.generate_synthetic_network(n_suppliers=100, n_manufacturers=30, n_retailers=50)
    
    # Initialize analyzer
    analyzer = SystemicRiskAnalyzer(test_network, verification_mode="strict")
    
    # Compute comprehensive risk metrics
    risk_metrics = analyzer.compute_comprehensive_risk_metrics()
    
    # Validate results
    validator = RiskMetricsValidator()
    validation_report = validator.validate_network_metrics(risk_metrics)
    
    # Save results
    file_paths = save_risk_analysis_results(risk_metrics, validation_report)
    
    print(f"Risk analysis completed for {len(risk_metrics)} nodes")
    print(f"Validation pass rate: {validation_report.get('validation_results', {}).get('bounds_check', {})}")