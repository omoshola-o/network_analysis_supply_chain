"""
Network Analysis for Systemic Risk Assessment in Supply Chains
Data Preprocessing Module

I created this module to generate realistic synthetic supply chain network data.
It handles all the data preprocessing tasks and includes validation checks
to ensure the data integrity throughout the analysis.

Author: Omoshola S. Owolabi
Date: 2024-12-21
"""

import numpy as np
import pandas as pd
import networkx as nx
import random
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplyChainDataGenerator:
    """
    My class for generating synthetic supply chain network data. I designed it to create
    realistic network structures with financial and operational characteristics that
    mirror real-world supply chains.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_synthetic_network(self, 
                                 n_suppliers: int = 500,
                                 n_manufacturers: int = 100,
                                 n_retailers: int = 200,
                                 connection_probability: float = 0.15) -> nx.DiGraph:
        """
        Generate synthetic supply chain network with realistic topology.
        
        Args:
            n_suppliers: Number of upstream suppliers
            n_manufacturers: Number of mid-tier manufacturers  
            n_retailers: Number of downstream retailers
            connection_probability: Probability of edge formation between tiers
            
        Returns:
            NetworkX directed graph representing supply chain
        """
        logger.info(f"Generating network with {n_suppliers} suppliers, {n_manufacturers} manufacturers, {n_retailers} retailers")
        
        G = nx.DiGraph()
        
        # Create node sets by tier
        suppliers = [f"S_{i:03d}" for i in range(n_suppliers)]
        manufacturers = [f"M_{i:03d}" for i in range(n_manufacturers)]
        retailers = [f"R_{i:03d}" for i in range(n_retailers)]
        
        all_nodes = suppliers + manufacturers + retailers
        
        # Add nodes with attributes
        for node in all_nodes:
            tier = self._get_node_tier(node)
            attributes = self._generate_node_attributes(node, tier)
            G.add_node(node, **attributes)
        
        # Create tier-based connections
        self._create_supply_connections(G, suppliers, manufacturers, connection_probability)
        self._create_supply_connections(G, manufacturers, retailers, connection_probability * 1.2)
        
        # Add some cross-tier connections for realism
        self._add_cross_tier_connections(G, suppliers, retailers, connection_probability * 0.1)
        
        logger.info(f"Generated network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _get_node_tier(self, node_id: str) -> str:
        """Extract tier from node ID."""
        return node_id.split('_')[0]
    
    def _generate_node_attributes(self, node_id: str, tier: str) -> Dict[str, Any]:
        """Generate realistic node attributes based on tier."""
        base_attrs = {
            'node_id': node_id,
            'tier': tier,
            'company_size': np.random.choice(['Small', 'Medium', 'Large'], p=[0.6, 0.3, 0.1]),
            'geographic_region': np.random.choice(['North America', 'Europe', 'Asia', 'Other'], p=[0.3, 0.25, 0.35, 0.1])
        }
        
        # Tier-specific attributes
        if tier == 'S':  # Suppliers
            base_attrs.update({
                'revenue_millions': np.random.lognormal(mean=2.0, sigma=1.5),
                'debt_to_equity': np.random.gamma(2, 0.5),
                'liquidity_ratio': np.random.gamma(2, 0.8),
                'working_capital_days': np.random.normal(45, 15),
                'supplier_diversification': np.random.poisson(3) + 1,
                'customer_concentration': np.random.beta(2, 5)
            })
        elif tier == 'M':  # Manufacturers
            base_attrs.update({
                'revenue_millions': np.random.lognormal(mean=3.5, sigma=1.2),
                'debt_to_equity': np.random.gamma(1.5, 0.6),
                'liquidity_ratio': np.random.gamma(2.5, 0.6),
                'working_capital_days': np.random.normal(30, 10),
                'supplier_diversification': np.random.poisson(8) + 2,
                'customer_concentration': np.random.beta(3, 7)
            })
        else:  # Retailers
            base_attrs.update({
                'revenue_millions': np.random.lognormal(mean=4.0, sigma=1.0),
                'debt_to_equity': np.random.gamma(1.8, 0.4),
                'liquidity_ratio': np.random.gamma(3, 0.5),
                'working_capital_days': np.random.normal(25, 8),
                'supplier_diversification': np.random.poisson(12) + 3,
                'customer_concentration': np.random.beta(1, 10)
            })
        
        # Ensure positive values
        for key in ['revenue_millions', 'debt_to_equity', 'liquidity_ratio', 'working_capital_days']:
            base_attrs[key] = max(0.1, base_attrs[key])
        
        return base_attrs
    
    def _create_supply_connections(self, G: nx.DiGraph, upstream: List[str], 
                                 downstream: List[str], prob: float):
        """Create connections between upstream and downstream tiers."""
        for up_node in upstream:
            for down_node in downstream:
                if np.random.random() < prob:
                    # Generate edge attributes
                    edge_attrs = self._generate_edge_attributes(G.nodes[up_node], G.nodes[down_node])
                    G.add_edge(up_node, down_node, **edge_attrs)
    
    def _add_cross_tier_connections(self, G: nx.DiGraph, suppliers: List[str], 
                                  retailers: List[str], prob: float):
        """Add direct supplier-to-retailer connections."""
        for supplier in suppliers:
            for retailer in retailers:
                if np.random.random() < prob:
                    edge_attrs = self._generate_edge_attributes(G.nodes[supplier], G.nodes[retailer])
                    G.add_edge(supplier, retailer, **edge_attrs)
    
    def _generate_edge_attributes(self, source_attrs: Dict, target_attrs: Dict) -> Dict[str, float]:
        """Generate edge attributes based on node characteristics."""
        # Transaction volume based on both nodes' revenue
        base_volume = np.sqrt(source_attrs['revenue_millions'] * target_attrs['revenue_millions'])
        transaction_volume = np.random.lognormal(np.log(base_volume), 0.5)
        
        return {
            'transaction_volume_millions': max(0.01, transaction_volume),
            'dependency_strength': np.random.beta(2, 3),
            'lead_time_days': np.random.gamma(2, 5),
            'payment_terms_days': np.random.choice([30, 45, 60, 90], p=[0.4, 0.3, 0.2, 0.1]),
            'contract_duration_months': np.random.choice([6, 12, 24, 36], p=[0.2, 0.4, 0.3, 0.1])
        }

class DataVerification:
    """
    Comprehensive data verification and integrity checking functions.
    """
    
    @staticmethod
    def verify_network_integrity(G: nx.DiGraph) -> Dict[str, Any]:
        """
        Verify network structure and data integrity.
        
        Returns:
            Dictionary containing verification results and diagnostics
        """
        verification_results = {
            'timestamp': datetime.now().isoformat(),
            'network_basic_stats': {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'is_directed': G.is_directed(),
                'is_connected': nx.is_weakly_connected(G)
            },
            'node_verification': {},
            'edge_verification': {},
            'data_quality_checks': {}
        }
        
        # Verify node attributes
        node_attrs = ['revenue_millions', 'debt_to_equity', 'liquidity_ratio', 
                     'working_capital_days', 'supplier_diversification', 'customer_concentration']
        
        for attr in node_attrs:
            values = [G.nodes[node].get(attr, 0) for node in G.nodes()]
            verification_results['node_verification'][attr] = {
                'count': len([v for v in values if v > 0]),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'has_negatives': any(v < 0 for v in values),
                'has_nulls': any(v is None or np.isnan(v) for v in values)
            }
        
        # Verify edge attributes
        edge_attrs = ['transaction_volume_millions', 'dependency_strength', 
                     'lead_time_days', 'payment_terms_days', 'contract_duration_months']
        
        for attr in edge_attrs:
            values = [G.edges[edge].get(attr, 0) for edge in G.edges()]
            verification_results['edge_verification'][attr] = {
                'count': len([v for v in values if v > 0]),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'has_negatives': any(v < 0 for v in values),
                'has_nulls': any(v is None or np.isnan(v) for v in values)
            }
        
        # Data quality checks
        verification_results['data_quality_checks'] = {
            'tier_distribution': DataVerification._check_tier_distribution(G),
            'connectivity_check': DataVerification._check_connectivity_patterns(G),
            'attribute_completeness': DataVerification._check_attribute_completeness(G),
            'logical_consistency': DataVerification._check_logical_consistency(G)
        }
        
        logger.info("Network integrity verification completed")
        return verification_results
    
    @staticmethod
    def _check_tier_distribution(G: nx.DiGraph) -> Dict[str, int]:
        """Check distribution of nodes across tiers."""
        tier_counts = {}
        for node in G.nodes():
            tier = G.nodes[node].get('tier', 'Unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        return tier_counts
    
    @staticmethod
    def _check_connectivity_patterns(G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze connectivity patterns between tiers."""
        tier_connections = {}
        for edge in G.edges():
            source_tier = G.nodes[edge[0]].get('tier', 'Unknown')
            target_tier = G.nodes[edge[1]].get('tier', 'Unknown')
            connection_type = f"{source_tier}_to_{target_tier}"
            tier_connections[connection_type] = tier_connections.get(connection_type, 0) + 1
        
        return {
            'tier_connections': tier_connections,
            'avg_out_degree': np.mean([G.out_degree(node) for node in G.nodes()]),
            'avg_in_degree': np.mean([G.in_degree(node) for node in G.nodes()])
        }
    
    @staticmethod
    def _check_attribute_completeness(G: nx.DiGraph) -> Dict[str, float]:
        """Check completeness of node and edge attributes."""
        expected_node_attrs = ['revenue_millions', 'debt_to_equity', 'liquidity_ratio', 
                              'working_capital_days', 'supplier_diversification', 'customer_concentration']
        expected_edge_attrs = ['transaction_volume_millions', 'dependency_strength', 
                              'lead_time_days', 'payment_terms_days', 'contract_duration_months']
        
        completeness = {}
        
        # Node attribute completeness
        for attr in expected_node_attrs:
            complete_count = sum(1 for node in G.nodes() if attr in G.nodes[node] and G.nodes[node][attr] is not None)
            completeness[f"node_{attr}"] = complete_count / G.number_of_nodes()
        
        # Edge attribute completeness
        for attr in expected_edge_attrs:
            complete_count = sum(1 for edge in G.edges() if attr in G.edges[edge] and G.edges[edge][attr] is not None)
            completeness[f"edge_{attr}"] = complete_count / G.number_of_edges() if G.number_of_edges() > 0 else 0
        
        return completeness
    
    @staticmethod
    def _check_logical_consistency(G: nx.DiGraph) -> Dict[str, bool]:
        """Check logical consistency of data relationships."""
        consistency_checks = {
            'positive_revenues': all(G.nodes[node].get('revenue_millions', 0) > 0 for node in G.nodes()),
            'valid_debt_ratios': all(G.nodes[node].get('debt_to_equity', 0) >= 0 for node in G.nodes()),
            'positive_liquidity': all(G.nodes[node].get('liquidity_ratio', 0) > 0 for node in G.nodes()),
            'positive_transaction_volumes': all(G.edges[edge].get('transaction_volume_millions', 0) > 0 for edge in G.edges()),
            'valid_dependency_strength': all(0 <= G.edges[edge].get('dependency_strength', 0) <= 1 for edge in G.edges())
        }
        
        return consistency_checks

def save_network_data(G: nx.DiGraph, base_path: str = "journal_package") -> Dict[str, str]:
    """
    Save network data in multiple formats for analysis and verification.
    
    Args:
        G: NetworkX graph to save
        base_path: Base directory path for saving files
        
    Returns:
        Dictionary with paths to saved files
    """
    file_paths = {}
    
    # Save as JSON for compatibility (GML has string value restrictions)
    json_path = f"multi_tier_supply_network.json"
    graph_data = nx.node_link_data(G)
    with open(json_path, 'w') as f:
        json.dump(graph_data, f, indent=2, default=str)
    file_paths['json'] = json_path
    
    # Save node data as CSV
    node_data = []
    for node in G.nodes():
        node_dict = {'node_id': node}
        node_dict.update(G.nodes[node])
        node_data.append(node_dict)
    
    nodes_df = pd.DataFrame(node_data)
    nodes_path = f"network_nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)
    file_paths['nodes_csv'] = nodes_path
    
    # Save edge data as CSV
    edge_data = []
    for edge in G.edges():
        edge_dict = {'source': edge[0], 'target': edge[1]}
        edge_dict.update(G.edges[edge])
        edge_data.append(edge_dict)
    
    edges_df = pd.DataFrame(edge_data)
    edges_path = f"network_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    file_paths['edges_csv'] = edges_path
    
    logger.info(f"Network data saved to: {list(file_paths.values())}")
    return file_paths

def load_and_validate_data(data_path: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Load network data and perform comprehensive validation.
    
    Args:
        data_path: Path to network data file
        
    Returns:
        Tuple of (loaded graph, validation results)
    """
    try:
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                graph_data = json.load(f)
            G = nx.node_link_graph(graph_data)
        else:
            G = nx.read_gml(data_path)
        validation_results = DataVerification.verify_network_integrity(G)
        logger.info(f"Successfully loaded and validated network from {data_path}")
        return G, validation_results
    except Exception as e:
        logger.error(f"Failed to load network data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage and testing
    generator = SupplyChainDataGenerator(seed=42)
    
    # Generate network
    network = generator.generate_synthetic_network(
        n_suppliers=300,
        n_manufacturers=80,
        n_retailers=120,
        connection_probability=0.12
    )
    
    # Verify network integrity
    verification_results = DataVerification.verify_network_integrity(network)
    
    # Save network and verification results
    file_paths = save_network_data(network)
    
    # Save verification results
    with open("verification_outputs/data_verification.json", "w") as f:
        json.dump(verification_results, f, indent=2, default=str)
    
    print(f"Generated network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    print(f"Data integrity verification completed. Results saved to verification_outputs/")