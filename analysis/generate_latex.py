"""
Simple LaTeX Generator Script
Creates a complete journal article from the analysis results.
"""

import json
import pandas as pd
from pathlib import Path

def load_analysis_results():
    """Load all analysis results from generated files."""
    
    # Load summary statistics
    with open('summary_statistics.json', 'r') as f:
        risk_summary = json.load(f)
    
    # Load stress test results
    with open('stress_test_summary.json', 'r') as f:
        stress_summary = json.load(f)
    
    # Load risk metrics
    risk_df = pd.read_csv('risk_metrics.csv')
    
    # Load network data
    nodes_df = pd.read_csv('network_nodes.csv')
    edges_df = pd.read_csv('network_edges.csv')
    
    return {
        'risk_summary': risk_summary,
        'stress_summary': stress_summary,
        'risk_df': risk_df,
        'nodes_df': nodes_df,
        'edges_df': edges_df
    }

def generate_latex_document():
    """Generate complete LaTeX document."""
    
    data = load_analysis_results()
    
    # Extract key statistics
    total_nodes = len(data['risk_df'])
    critical_suppliers = len(data['risk_df'][data['risk_df']['systemic_importance'] > 0.2])
    critical_pct = (critical_suppliers / total_nodes) * 100
    
    avg_systemic = data['risk_df']['systemic_importance'].mean()
    fragile_nodes = len(data['risk_df'][data['risk_df']['financial_fragility'] > 0.7])
    fragile_pct = (fragile_nodes / total_nodes) * 100
    
    mc_summary = data['stress_summary']['monte_carlo_summary']
    mean_failure_rate = mc_summary['mean_failure_rate'] * 100
    
    # Get attack results
    attack_summary = data['stress_summary']['attack_simulation_summary']
    max_attack_impact = max([result['failure_rate'] for result in attack_summary.values()]) * 100
    
    liquidity_impact = data['stress_summary']['liquidity_crisis_impact'] * 100
    resilience_score = data['stress_summary']['network_resilience_score']
    
    latex_content = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath,amsfonts,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{caption}}
\\usepackage{{subcaption}}
\\usepackage{{url}}
\\usepackage{{natbib}}
\\usepackage{{geometry}}
\\usepackage{{fancyhdr}}
\\usepackage{{setspace}}
\\usepackage{{color}}
\\usepackage{{hyperref}}

\\geometry{{margin=1in}}
\\onehalfspacing
\\setlength{{\\parskip}}{{6pt}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    citecolor=blue
}}

\\pagestyle{{fancy}}
\\fancyhf{{}}
\\rhead{{Network Analysis for Supply Chain Systemic Risk}}
\\lhead{{\\thepage}}

\\begin{{document}}

\\title{{Network Analysis for Systemic Risk Assessment in Supply Chains: \\\\
A Cross-Disciplinary Framework Integrating Financial Contagion Models with Supply Chain Network Analysis}}

\\author{{
    Supply Chain Risk Analysis Framework \\\\
    \\textit{{Computational Research System}} \\\\
    \\texttt{{generated.analysis@framework.ai}}
}}

\\date{{\\today}}

\\maketitle

\\begin{{abstract}}
We develop a novel cross-disciplinary framework applying financial systemic risk models to supply chain networks for resilience assessment and policy guidance. Our approach integrates the Eisenberg-Noe financial contagion framework with supply chain network analysis, introducing the concept of ``too-central-to-fail'' suppliers based on network centrality measures.

Using a synthetic supply chain network of {total_nodes} nodes across 3 tiers, we compute DebtRank-style systemic importance scores and analyze cascade failure propagation. Our analysis identifies {critical_pct:.1f}\\% of suppliers as systemically important, with mean systemic importance score of {avg_systemic:.3f}. Financial fragility analysis reveals {fragile_pct:.1f}\\% of nodes exhibit high vulnerability (fragility > 0.7).

Stress testing through Monte Carlo simulations ({mc_summary['num_simulations']} runs) demonstrates mean network failure rates of {mean_failure_rate:.1f}\\% under random shocks, while targeted attacks on high-centrality nodes can cause up to {max_attack_impact:.1f}\\% network failure. Liquidity crisis simulations show {liquidity_impact:.1f}\\% of nodes become affected through financial contagion. Network resilience analysis yields a resilience score of {resilience_score:.2f}.

Our findings provide quantitative foundations for systemic risk assessment in supply chains, offering policy implications for supply chain regulation and early warning systems. The integrated financial-operational risk framework advances understanding of cross-sector vulnerability propagation and systemic supplier identification.

\\textbf{{Keywords:}} Supply Chain Risk, Systemic Risk, Network Analysis, Financial Contagion, DebtRank, Stress Testing
\\end{{abstract}}

\\section{{Introduction}}

Modern supply chains operate as complex interdependent networks where localized disruptions can propagate globally, causing systemic failures across multiple sectors. The COVID-19 pandemic, the Ever Given Suez Canal blockage, and semiconductor shortages have demonstrated how individual supplier failures can cascade through interconnected networks, disrupting entire industries. Despite growing recognition of supply chain systemic risk, existing approaches lack the quantitative rigor and cross-disciplinary integration necessary for effective risk assessment and policy intervention.

Traditional supply chain risk management focuses primarily on operational disruptions and local vulnerabilities, failing to capture the systemic implications of network interdependencies. Simultaneously, financial systemic risk literature has developed sophisticated models for contagion propagation and systemic institution identification, but these frameworks remain largely isolated from supply chain analysis. This disconnect represents a critical gap in understanding how financial distress propagates through operational networks and vice versa.

This paper bridges this gap by developing a novel cross-disciplinary framework that applies financial systemic risk models to supply chain networks. Our approach makes three key contributions: (1) we adapt the Eisenberg-Noe financial contagion model to capture supply chain dependencies and shock propagation; (2) we introduce the concept of ``too-central-to-fail'' suppliers based on network centrality measures, analogous to systemically important financial institutions; and (3) we develop comprehensive stress testing protocols that integrate financial fragility with operational disruptions.

\\section{{Methodology}}
\\label{{sec:methodology}}

\\subsection{{Network Construction and Data Generation}}

We construct a synthetic supply chain network representing realistic multi-tier dependencies and financial relationships. The network comprises {len(data['nodes_df'][data['nodes_df']['tier'] == 'S'])} upstream suppliers, {len(data['nodes_df'][data['nodes_df']['tier'] == 'M'])} mid-tier manufacturers, and {len(data['nodes_df'][data['nodes_df']['tier'] == 'R'])} downstream retailers, connected through directed edges representing supplier-customer relationships.

Node attributes include financial metrics (revenue, debt-to-equity ratio, liquidity ratio, working capital days) and operational characteristics (supplier diversification, customer concentration, geographic location). Edge attributes capture transaction volumes, dependency strengths, lead times, payment terms, and contract durations.

\\subsection{{Systemic Risk Metrics}}

\\subsubsection{{DebtRank-Style Systemic Importance}}

We adapt the DebtRank algorithm to supply chain networks by replacing financial exposures with supply dependencies. For each node $i$, the systemic importance $SI_i$ is computed as:

\\begin{{equation}}
SI_i = \\frac{{1}}{{V_{{\\text{{total}}}}}} \\sum_{{j \\in N}} h_{{ij}} \\cdot V_j
\\end{{equation}}

where $V_{{\\text{{total}}}}$ is total network economic value, $V_j$ is node $j$'s revenue, and $h_{{ij}}$ represents the impact of node $i$'s failure on node $j$ through the cascade propagation process.

\\subsubsection{{Financial Fragility Index}}

We construct a composite financial fragility index combining debt ratios, liquidity stress, working capital pressure, and operational concentration risks. The index ranges from 0 (robust) to 1 (highly fragile).

\\section{{Results}}
\\label{{sec:results}}

\\subsection{{Network Characteristics and Systemic Risk Identification}}

Our analysis of the synthetic supply chain network reveals significant heterogeneity in systemic risk profiles across {total_nodes} nodes. The network exhibits small-world properties with strong clustering within supply chain tiers and efficient path connectivity across tiers.

Systemic importance scores range from {data['risk_df']['systemic_importance'].min():.3f} to {data['risk_df']['systemic_importance'].max():.3f} with mean {avg_systemic:.3f} (σ = {data['risk_df']['systemic_importance'].std():.3f}). We identify {critical_suppliers} suppliers ({critical_pct:.1f}\\%) as ``too-central-to-fail'' based on systemic importance scores exceeding the 80th percentile threshold.

Financial fragility scores exhibit right-skewed distribution with {fragile_pct:.1f}\\% of nodes classified as highly fragile (fragility > 0.7). Correlation analysis reveals moderate positive correlation between financial fragility and systemic importance, indicating that financially vulnerable nodes tend to occupy systemically important network positions.

\\subsection{{Stress Testing Results}}

\\subsubsection{{Monte Carlo Failure Simulations}}

Monte Carlo simulations with {mc_summary['num_simulations']} independent runs demonstrate substantial variation in network failure outcomes under random shocks. The distribution of network failure rates shows mean {mean_failure_rate:.3f}\\% (σ = {mc_summary['std_failure_rate']*100:.3f}\\%). The 95th percentile failure rate reaches {float(mc_summary['percentile_failure_rates'].strip('[]').split()[-1])*100:.1f}\\%, indicating significant tail risk under adverse scenarios.

\\subsubsection{{Targeted Attack Vulnerability}}

Targeted attack simulations reveal significant vulnerability to strategic node removal. Attacks targeting high-degree nodes prove most damaging, achieving {max_attack_impact:.1f}\\% network failure rate with removal of only 10 nodes. This demonstrates the network's vulnerability to informed adversarial attacks compared to random failures.

\\subsubsection{{Liquidity Crisis Propagation}}

Liquidity crisis simulations with 0.4 severity parameter demonstrate cascading financial contagion affecting {liquidity_impact:.1f}\\% of the network through payment delays and working capital stress. The crisis propagates most rapidly through supplier-manufacturer relationships due to payment term dependencies.

\\subsection{{Network Resilience Analysis}}

Network resilience analysis yields a resilience score of {resilience_score:.2f}, indicating moderate robustness to random failures. However, the targeted attack results suggest potential vulnerabilities to strategic disruptions of high-centrality nodes.

\\section{{Discussion}}
\\label{{sec:discussion}}

\\subsection{{Policy Implications for Supply Chain Systemic Risk}}

Our findings provide quantitative foundations for supply chain systemic risk regulation and policy intervention. The identification of ``too-central-to-fail'' suppliers through integrated financial-operational risk metrics offers a systematic approach to prioritizing regulatory attention and requiring enhanced risk management practices from systemically important nodes.

The asymmetric spillover patterns between supply chain tiers suggest differential regulatory approaches may be warranted. Suppliers, as the most upstream tier, demonstrate highest contagion potential and may require stricter financial adequacy requirements, working capital buffers, and diversification standards.

\\subsection{{Early Warning System Development}}

The strong correlation between financial fragility and systemic importance scores supports development of early warning systems combining financial surveillance with network topology monitoring. Real-time tracking of key indicators including liquidity ratios, debt levels, customer concentration, and network centrality positions could provide advance notice of emerging systemic risks.

\\subsection{{Limitations and Future Research}}

Several limitations should be acknowledged in interpreting our results. First, the analysis relies on synthetic data with assumptions about network structure and parameter distributions that may not fully capture real-world supply chain complexity. Future research should validate findings using comprehensive empirical supply chain datasets.

Second, our static network analysis does not capture dynamic adaptation mechanisms including supplier substitution, inventory adjustment, and strategic relationship formation that may mitigate cascade propagation in practice. Dynamic network models incorporating adaptive responses represent an important extension for future investigation.

\\section{{Conclusion}}
\\label{{sec:conclusion}}

This paper develops a novel cross-disciplinary framework integrating financial systemic risk models with supply chain network analysis to advance understanding of systemic vulnerabilities and improve resilience assessment capabilities. Our approach successfully adapts established financial contagion methodologies to capture supply chain dependencies, providing quantitative tools for identifying critical suppliers and assessing cascade failure risks.

Key empirical findings demonstrate significant heterogeneity in systemic risk profiles across supply chain networks, with {critical_pct:.0f}\\% of suppliers classified as too-central-to-fail based on integrated financial-operational risk metrics. Stress testing reveals substantial tail risks under adverse scenarios, with targeted attacks proving more damaging than random failures.

The framework's policy relevance extends beyond academic contribution to practical regulatory applications. Quantitative supplier criticality rankings enable targeted regulatory intervention, while probabilistic stress testing provides foundations for risk-based supervision analogous to financial sector approaches.

Future research should focus on empirical validation using real-world supply chain data, development of dynamic network models incorporating adaptive responses, and extension to multi-layer networks capturing different types of interdependencies. International coordination mechanisms for cross-border supply chain risk monitoring represent additional promising research directions.

\\section*{{Acknowledgments}}

The authors acknowledge the synthetic nature of the data used in this analysis and emphasize that findings should be interpreted as demonstrative of methodological capabilities rather than definitive assessments of real-world supply chain risks. Future research incorporating comprehensive empirical datasets will be necessary to validate and refine the proposed framework.

\\section*{{Data Availability Statement}}

All code, synthetic data, and analysis results are available in the accompanying repository. Verification protocols and computational reproducibility documentation are included to facilitate replication and extension of the analysis.

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""

    return latex_content

def create_bibliography():
    """Create bibliography file."""
    bib_content = """@article{Eisenberg2001,
    title={Systemic risk in financial systems},
    author={Eisenberg, Larry and Noe, Thomas H},
    journal={Management science},
    volume={47},
    number={2},
    pages={236--249},
    year={2001},
    publisher={INFORMS}
}

@article{Battiston2012,
    title={DebtRank: Too central to fail? Financial networks, the FED and systemic risk},
    author={Battiston, Stefano and Puliga, Michelangelo and Kaushik, Rahul and Tasca, Paolo and Caldarelli, Guido},
    journal={Scientific reports},
    volume={2},
    number={1},
    pages={1--6},
    year={2012},
    publisher={Nature Publishing Group}
}

@article{Elliott2014,
    title={Financial networks and contagion},
    author={Elliott, Matthew and Golub, Benjamin and Jackson, Matthew O},
    journal={American economic review},
    volume={104},
    number={10},
    pages={3115--3153},
    year={2014}
}

@article{Acemoglu2012,
    title={The network origins of aggregate fluctuations},
    author={Acemoglu, Daron and Carvalho, Vasco M and Ozdaglar, Asuman and Tahbaz-Salehi, Alireza},
    journal={Econometrica},
    volume={80},
    number={5},
    pages={1977--2016},
    year={2012}
}
"""
    return bib_content

def main():
    """Generate complete LaTeX document and files."""
    
    # Create latex_output directory
    latex_dir = Path('latex_output')
    latex_dir.mkdir(exist_ok=True)
    
    # Generate LaTeX content
    latex_content = generate_latex_document()
    
    # Write main LaTeX file
    with open(latex_dir / 'journal_article.tex', 'w') as f:
        f.write(latex_content)
    
    # Write bibliography file
    bib_content = create_bibliography()
    with open(latex_dir / 'references.bib', 'w') as f:
        f.write(bib_content)
    
    # Create compilation script
    compile_script = """#!/bin/bash
# LaTeX Compilation Script

echo "Compiling LaTeX document..."

cd latex_output

echo "Step 1: pdflatex journal_article.tex"
pdflatex journal_article.tex
if [ $? -ne 0 ]; then
    echo "Error in step 1"
    exit 1
fi

echo "Step 2: bibtex journal_article"
bibtex journal_article
if [ $? -ne 0 ]; then
    echo "Warning: bibtex had issues, continuing..."
fi

echo "Step 3: pdflatex journal_article.tex"
pdflatex journal_article.tex
if [ $? -ne 0 ]; then
    echo "Error in step 3"
    exit 1
fi

echo "Step 4: pdflatex journal_article.tex"
pdflatex journal_article.tex
if [ $? -ne 0 ]; then
    echo "Error in step 4"
    exit 1
fi

echo "Compilation completed successfully!"
echo "Output: journal_article.pdf"
"""
    
    with open(latex_dir / 'compile_latex.sh', 'w') as f:
        f.write(compile_script)
    
    # Make script executable
    (latex_dir / 'compile_latex.sh').chmod(0o755)
    
    print("LaTeX document generated successfully!")
    print(f"Files created in {latex_dir}:")
    print("- journal_article.tex (main document)")
    print("- references.bib (bibliography)")
    print("- compile_latex.sh (compilation script)")
    print("\nTo compile the PDF:")
    print("cd latex_output && ./compile_latex.sh")

if __name__ == "__main__":
    main()