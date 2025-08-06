import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
import sys

# Add parent directory to path to import database and models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database import engine

# Add models path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sqlite-scripts'))
from models import *

# Create assets directory
assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(assets_dir, exist_ok=True)

Session = sessionmaker(bind=engine)

def load_patient_data(max_samples=None):
    """Load and aggregate patient data from multiple tables"""
    print("Loading patient data...")
    
    # Load basic patient demographics
    patients_query = """
    SELECT p.PatientID, p.Name, p.DOB, p.Gender, p.Address,
           CASE 
               WHEN p.DOB IS NOT NULL THEN 
                   (julianday('now') - julianday(p.DOB)) / 365.25
               ELSE NULL 
           END as Age
    FROM Patients p
    """
    patients_df = pd.read_sql(patients_query, engine)
    if max_samples is not None:
        patients_df = patients_df.head(max_samples)
    
    # Aggregate appointment data
    appointments_query = """
    SELECT PatientID, 
           COUNT(*) as appointment_count,
           COUNT(DISTINCT ProfessionalID) as unique_professionals,
           COUNT(DISTINCT DepartmentID) as unique_departments
    FROM Appointments
    GROUP BY PatientID
    """
    appointments_df = pd.read_sql(appointments_query, engine)
    
    # Aggregate prescription data
    prescriptions_query = """
    SELECT p.PatientID,
           COUNT(DISTINCT pr.PrescriptionID) as prescription_count,
           COUNT(DISTINCT pd.MedicationID) as unique_medications,
           AVG(pd.TotalBillingAmount) as avg_prescription_cost,
           SUM(pd.TotalBillingAmount) as total_prescription_cost
    FROM Patients p
    JOIN MedicalRecords mr ON p.PatientID = mr.PatientID
    JOIN Prescriptions pr ON mr.RecordID = pr.RecordID
    JOIN PrescriptionDetails pd ON pr.PrescriptionID = pd.PrescriptionID
    GROUP BY p.PatientID
    """
    prescriptions_df = pd.read_sql(prescriptions_query, engine)
    
    # Aggregate test data
    tests_query = """
    SELECT PatientID,
           COUNT(*) as test_count,
           COUNT(DISTINCT TestName) as unique_test_types
    FROM Tests
    GROUP BY PatientID
    """
    tests_df = pd.read_sql(tests_query, engine)
    
    # Aggregate surgery data
    surgeries_query = """
    SELECT PatientID,
           COUNT(*) as surgery_count,
           COUNT(DISTINCT Type) as unique_surgery_types
    FROM Surgeries
    GROUP BY PatientID
    """
    surgeries_df = pd.read_sql(surgeries_query, engine)
    
    # Aggregate billing data
    billing_query = """
    SELECT PatientID,
           COUNT(*) as billing_count,
           AVG(Amount) as avg_billing_amount,
           SUM(Amount) as total_billing_amount,
           AVG(AmountPaid) as avg_amount_paid,
           SUM(AmountPaid) as total_amount_paid
    FROM ServiceBillings
    GROUP BY PatientID
    """
    billing_df = pd.read_sql(billing_query, engine)
    
    # Merge all dataframes
    df = patients_df.copy()
    df = df.merge(appointments_df, on='PatientID', how='left')
    df = df.merge(prescriptions_df, on='PatientID', how='left')
    df = df.merge(tests_df, on='PatientID', how='left')
    df = df.merge(surgeries_df, on='PatientID', how='left')
    df = df.merge(billing_df, on='PatientID', how='left')
    
    # Fill NaN values with 0 for count and sum columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"Loaded data for {len(df)} patients with {len(df.columns)} features")
    return df

def prepare_features(df):
    """Prepare feature matrix for clustering"""
    print("Preparing features for clustering...")
    
    # Select numerical features for clustering
    feature_cols = [
        'Age', 'appointment_count', 'unique_professionals', 'unique_departments',
        'prescription_count', 'unique_medications', 'avg_prescription_cost',
        'total_prescription_cost', 'test_count', 'unique_test_types',
        'surgery_count', 'unique_surgery_types', 'billing_count',
        'avg_billing_amount', 'total_billing_amount', 'avg_amount_paid',
        'total_amount_paid'
    ]
    
    # Create feature matrix
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Add gender encoding
    le_gender = LabelEncoder()
    gender_encoded = le_gender.fit_transform(df['Gender'].fillna('Unknown'))
    X['gender_encoded'] = gender_encoded
    
    # Create utilization ratios
    X['prescription_per_appointment'] = X['prescription_count'] / (X['appointment_count'] + 1)
    X['test_per_appointment'] = X['test_count'] / (X['appointment_count'] + 1)
    X['payment_ratio'] = X['total_amount_paid'] / (X['total_billing_amount'] + 1)
    
    print(f"Prepared {X.shape[1]} features for {X.shape[0]} patients")
    return X, feature_cols

def perform_pca_analysis(X_scaled):
    """Perform PCA for dimensionality reduction"""
    print("Performing PCA analysis...")
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Find optimal number of components (95% variance)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    
    # Plot PCA variance explanation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'bo-')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA - Individual Component Variance')
    ax1.grid(True)
    
    ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'ro-')
    ax2.axhline(y=0.95, color='k', linestyle='--', label='95% Variance')
    ax2.axvline(x=n_components_95, color='k', linestyle='--', label=f'{n_components_95} Components')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('PCA - Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reduce to optimal dimensions
    pca_optimal = PCA(n_components=n_components_95)
    X_pca_reduced = pca_optimal.fit_transform(X_scaled)
    
    print(f"Reduced dimensions from {X_scaled.shape[1]} to {n_components_95} components")
    return X_pca_reduced, pca_optimal

def find_optimal_clusters(X, max_clusters=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    print("Finding optimal number of clusters...")
    
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    K_range = range(2, max_clusters + 1)

    for idx, k in enumerate(K_range):
        print(f"Evaluating k={idx} / {len(K_range)}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
    
    # Plot clustering metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True)
    
    ax3.plot(K_range, calinski_scores, 'go-')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Calinski-Harabasz Score')
    ax3.set_title('Calinski-Harabasz Score vs Number of Clusters')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'cluster_optimization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find optimal k (highest silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    return optimal_k

def perform_kmeans_clustering(X, optimal_k):
    """Perform K-means clustering"""
    print(f"Performing K-means clustering with {optimal_k} clusters...")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    return cluster_labels, kmeans

def perform_hierarchical_clustering(X, n_clusters):
    """Perform hierarchical clustering"""
    print("Performing hierarchical clustering...")
    
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    hierarchical_labels = hierarchical.fit_predict(X)
    
    return hierarchical_labels

def perform_dbscan_clustering(X):
    """Perform DBSCAN clustering to identify outliers"""
    print("Performing DBSCAN clustering...")
    
    # Try different eps values to find optimal one
    eps_values = np.arange(0.5, 3.0, 0.1)
    best_eps = 1.0
    best_score = -1
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        if len(set(labels)) > 1 and -1 in labels:  # Has both clusters and outliers
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                core_samples = labels != -1
                if np.sum(core_samples) > 5:
                    score = silhouette_score(X[core_samples], labels[core_samples])
                    if score > best_score:
                        best_score = score
                        best_eps = eps
    
    # Final DBSCAN with best parameters
    dbscan = DBSCAN(eps=best_eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_outliers = np.sum(dbscan_labels == -1)
    
    print(f"DBSCAN found {n_clusters} clusters and {n_outliers} outliers with eps={best_eps}")
    
    return dbscan_labels

def analyze_clusters(df, cluster_labels, feature_cols):
    """Analyze cluster characteristics"""
    print("Analyzing cluster characteristics...")
    
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = df_clustered.groupby('cluster')[feature_cols + ['Age']].agg(['mean', 'std', 'count']).round(2)
    
    # Save cluster statistics
    cluster_stats.to_csv(os.path.join(assets_dir, 'cluster_statistics.csv'))
    
    # Create cluster visualization
    n_clusters = len(np.unique(cluster_labels))
    
    # Select top features for visualization
    top_features = ['Age', 'appointment_count', 'prescription_count', 'test_count', 'total_billing_amount']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features):
        if i < len(axes):
            df_clustered.boxplot(column=feature, by='cluster', ax=axes[i])
            axes[i].set_title(f'{feature} by Cluster')
            axes[i].set_xlabel('Cluster')
    
    # Remove empty subplots
    for i in range(len(top_features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Cluster Characteristics Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'cluster_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_clustered, cluster_stats

def medication_association_analysis():
    """Perform association rule mining for medication patterns"""
    print("Performing medication association analysis...")
    
    # Get prescription data
    prescription_query = """
    SELECT pr.PrescriptionID, m.Name as MedicationName
    FROM Prescriptions pr
    JOIN PrescriptionDetails pd ON pr.PrescriptionID = pd.PrescriptionID
    JOIN Medications m ON pd.MedicationID = m.MedicationID
    """
    prescription_df = pd.read_sql(prescription_query, engine)
    
    if len(prescription_df) == 0:
        print("No prescription data available for association analysis")
        return
    
    # Create transaction data
    transactions = prescription_df.groupby('PrescriptionID')['MedicationName'].apply(list).tolist()
    
    # Apply transaction encoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_transactions, min_support=0.05, use_colnames=True)
    
    if len(frequent_itemsets) > 0:
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
        
        # Save results
        frequent_itemsets.to_csv(os.path.join(assets_dir, 'frequent_medication_patterns.csv'), index=False)
        rules.to_csv(os.path.join(assets_dir, 'medication_association_rules.csv'), index=False)
        
        # Visualize top rules
        if len(rules) > 0:
            top_rules = rules.nlargest(10, 'confidence')
            
            plt.figure(figsize=(12, 8))
            plt.scatter(top_rules['support'], top_rules['confidence'], 
                       s=top_rules['lift']*20, alpha=0.7)
            plt.xlabel('Support')
            plt.ylabel('Confidence')
            plt.title('Top Medication Association Rules\n(Bubble size represents lift)')
            
            for i, row in top_rules.iterrows():
                plt.annotate(f"{list(row['antecedents'])[0]} -> {list(row['consequents'])[0]}", 
                           (row['support'], row['confidence']), fontsize=8)
            
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(assets_dir, 'medication_association_rules.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Found {len(frequent_itemsets)} frequent medication patterns and {len(rules)} association rules")
    else:
        print("No frequent medication patterns found with current thresholds")

def network_analysis():
    """Perform network analysis of patient-provider relationships"""
    print("Performing network analysis...")
    
    # Get patient-professional relationships
    network_query = """
    SELECT p.PatientID, pr.ProfessionalID, pr.Name as ProfessionalName, 
           pr.Role, COUNT(*) as interaction_count
    FROM Appointments a
    JOIN Patients p ON a.PatientID = p.PatientID
    JOIN Professionals pr ON a.ProfessionalID = pr.ProfessionalID
    GROUP BY p.PatientID, pr.ProfessionalID, pr.Name, pr.Role
    """
    network_df = pd.read_sql(network_query, engine)
    
    if len(network_df) == 0:
        print("No network data available")
        return None, None, None
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes and edges
    for _, row in network_df.iterrows():
        G.add_node(f"P_{row['PatientID']}", type='patient')
        G.add_node(f"PR_{row['ProfessionalID']}", type='professional', 
                  role=row['Role'], name=row['ProfessionalName'])
        G.add_edge(f"P_{row['PatientID']}", f"PR_{row['ProfessionalID']}", 
                  weight=row['interaction_count'])
    
    # Calculate network metrics
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Analyze professional centrality
    centrality = nx.degree_centrality(G)
    professional_centrality = {node: centrality[node] for node in G.nodes() 
                             if node.startswith('PR_')}
    # Top 10 professionals by centrality
    top_professionals = sorted(professional_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_professionals_info = []
    for node, cent in top_professionals:
        data = G.nodes[node]
        top_professionals_info.append({
            'ProfessionalID': node.replace('PR_', ''),
            'Name': data.get('name', ''),
            'Role': data.get('role', ''),
            'Centrality': cent
        })
    top_professionals_df = pd.DataFrame(top_professionals_info)
    top_professionals_df.to_csv(os.path.join(assets_dir, 'top_professionals_by_centrality.csv'), index=False)
    
    # Improved network visualization
    if G.number_of_nodes() > 1000:
        # Sample network for visualization
        sample_nodes = list(G.nodes())[:200]
        G_sample = G.subgraph(sample_nodes)
    else:
        G_sample = G
    
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G_sample, k=1, iterations=50, seed=42)
    
    # Draw patients and professionals differently, scale by centrality
    patient_nodes = [n for n in G_sample.nodes() if n.startswith('P_')]
    professional_nodes = [n for n in G_sample.nodes() if n.startswith('PR_')]
    patient_sizes = [300 for _ in patient_nodes]
    professional_sizes = [1000 * centrality.get(n, 0.01) + 100 for n in professional_nodes]
    
    # Draw patients (circles, blue)
    nx.draw_networkx_nodes(G_sample, pos, nodelist=patient_nodes, 
                          node_color='lightblue', node_size=patient_sizes, alpha=0.7, node_shape='o', label='Patients')
    # Draw professionals (squares, orange)
    nx.draw_networkx_nodes(G_sample, pos, nodelist=professional_nodes, 
                          node_color='orange', node_size=professional_sizes, alpha=0.85, node_shape='s', label='Professionals')
    # Draw edges
    nx.draw_networkx_edges(G_sample, pos, alpha=0.3, width=0.7)
    # Draw top professional labels
    labels = {n: G.nodes[n]['name'] for n in professional_nodes if n in [x[0] for x in top_professionals]}
    nx.draw_networkx_labels(G_sample, pos, labels=labels, font_size=9, font_color='black')
    
    plt.title(f'Patient-Professional Network\n({len(patient_nodes)} patients, {len(professional_nodes)} professionals)', fontsize=16)
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'patient_professional_network.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save centrality analysis
    centrality_df = pd.DataFrame([
        {'NodeID': node, 'Centrality': centrality[node], 'Type': 'Professional' if node.startswith('PR_') else 'Patient'}
        for node in G.nodes()
    ])
    centrality_df.to_csv(os.path.join(assets_dir, 'network_centrality.csv'), index=False)
    
    print("Network analysis completed")
    # Return stats for report
    return G, centrality, top_professionals_df

def create_cluster_dashboard(df_clustered, X_pca, cluster_labels, assets_dir, max_samples=None):
    """Create interactive dashboard for cluster visualization and embed plots in HTML report"""
    print("Creating interactive cluster dashboard...")
    
    # Create 2D PCA plot for visualization
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_pca)
    
    # Create interactive scatter plot
    fig = px.scatter(
        x=X_pca_2d[:, 0], y=X_pca_2d[:, 1],
        color=cluster_labels.astype(str),
        title="Patient Clusters in PCA Space",
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
        hover_data={'Patient_ID': df_clustered['PatientID'][:len(X_pca_2d)]}
    )
    
    fig.update_layout(
        width=800, height=600,
        title_font_size=16
    )
    
    fig.write_html(os.path.join(assets_dir, 'cluster_dashboard.html'))
    
    # Create cluster summary report
    cluster_summary = df_clustered.groupby('cluster').agg({
        'Age': ['mean', 'std'],
        'appointment_count': ['mean', 'std'], 
        'prescription_count': ['mean', 'std'],
        'total_billing_amount': ['mean', 'std'],
        'PatientID': 'count'
    }).round(2)
    
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns]
    cluster_summary = cluster_summary.rename(columns={'PatientID_count': 'patient_count'})
    
    # Add network stats and top professionals to report
    G, centrality, top_professionals_df = network_analysis()
    network_stats_html = ""
    top_professionals_html = ""
    if G is not None:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        n_patients = len([n for n in G.nodes() if n.startswith('P_')])
        n_professionals = len([n for n in G.nodes() if n.startswith('PR_')])
        avg_degree = np.mean([d for n, d in G.degree()])
        density = nx.density(G)
        network_stats_html = f"""
        <h2>Patient-Professional Network Analysis</h2>
        <ul>
            <li><strong>Total Nodes:</strong> {n_nodes} (Patients: {n_patients}, Professionals: {n_professionals})</li>
            <li><strong>Total Edges (Interactions):</strong> {n_edges}</li>
            <li><strong>Average Degree:</strong> {avg_degree:.2f}</li>
            <li><strong>Network Density:</strong> {density:.4f}</li>
        </ul>
        """
        # Add top professionals table from CSV (for robustness, reload from file)
        top_prof_csv = os.path.join(assets_dir, 'top_professionals_by_centrality.csv')
        if os.path.exists(top_prof_csv):
            top_prof_df = pd.read_csv(top_prof_csv)
            top_professionals_html = "<h3>Top Professionals by Centrality</h3>"
            top_professionals_html += top_prof_df[['Name', 'Role', 'Centrality']].to_html(index=False)
        elif top_professionals_df is not None and not top_professionals_df.empty:
            top_professionals_html = "<h3>Top Professionals by Centrality</h3>"
            top_professionals_html += top_professionals_df[['Name', 'Role', 'Centrality']].to_html(index=False)
    else:
        network_stats_html = "<h2>Patient-Professional Network Analysis</h2><p><em>No network data available.</em></p>"
        top_professionals_html = ""

    # Embed images as base64 in HTML
    import base64
    def img_to_base64(path):
        with open(path, "rb") as img_f:
            return base64.b64encode(img_f.read()).decode("utf-8")

    img_files = [
        ('PCA Analysis', 'pca_analysis.png'),
        ('Cluster Optimization', 'cluster_optimization.png'),
        ('Cluster Characteristics', 'cluster_characteristics.png'),
        ('Patient-Professional Network', 'patient_professional_network.png')
    ]
    img_tags = ""
    for title, fname in img_files:
        fpath = os.path.join(assets_dir, fname)
        if os.path.exists(fpath):
            img_b64 = img_to_base64(fpath)
            img_tags += f'<h3>{title}</h3><img src="data:image/png;base64,{img_b64}" style="max-width:100%;margin-bottom:30px;"><br>'
        else:
            img_tags += f'<h3>{title}</h3><p><em>Image not available</em></p>'

    # Create HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Solution 3: NHS Patient Segmentation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2E86AB; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; border-left: 4px solid #2E86AB; }}
            img {{ border: 1px solid #ccc; padding: 4px; background: #fff; }}
        </style>
    </head>
    <body>
        <h1>Solution 3: NHS Patient Segmentation Analysis Report</h1>
        <div class="summary">
            <h2>Executive Summary</h2>
            <p><strong>Total Patients Analyzed:</strong> {len(df_clustered):,}</p>
            <p><strong>Number of Clusters Identified:</strong> {len(df_clustered['cluster'].unique())}</p>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {"<p><strong>Sample Limit:</strong> " + str(max_samples) + "</p>" if max_samples else ""}
        </div>
        
        <h2>Cluster Characteristics</h2>
        {cluster_summary.to_html()}
        
        <h2>Key Insights</h2>
        <ul>
            <li>Patients have been segmented into {len(df_clustered['cluster'].unique())} distinct groups based on healthcare utilization patterns</li>
            <li>Age and service utilization are key differentiating factors between clusters</li>
            <li>Each cluster represents a different patient archetype requiring tailored healthcare strategies</li>
        </ul>
        
        <h2>Recommendations</h2>
        <ul>
            <li>Develop targeted care programs for each patient segment</li>
            <li>Optimize resource allocation based on cluster characteristics</li>
            <li>Monitor cluster evolution over time to adapt strategies</li>
        </ul>
        {network_stats_html}
        {top_professionals_html}
        <h2>Visualizations</h2>
        {img_tags}
    </body>
    </html>
    """
    
    with open(os.path.join(assets_dir, 'report.html'), 'w') as f:
        f.write(html_report)
    
    print("Interactive dashboard and report created")

def main(max_samples=None):
    """Main execution function"""
    print("=" * 60)
    print("NHS PATIENT SEGMENTATION AND HEALTHCARE SERVICE OPTIMIZATION")
    print("=" * 60)
    
    try:
        # 1. Load and prepare data
        df = load_patient_data(max_samples=max_samples)
        X, feature_cols = prepare_features(df)
        
        # 2. Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 3. Dimensionality reduction
        X_pca, pca = perform_pca_analysis(X_scaled)
        
        # 4. Find optimal clusters
        optimal_k = find_optimal_clusters(X_pca)
        
        # 5. Perform clustering
        kmeans_labels, kmeans_model = perform_kmeans_clustering(X_pca, optimal_k)
        hierarchical_labels = perform_hierarchical_clustering(X_pca, optimal_k)
        dbscan_labels = perform_dbscan_clustering(X_pca)
        
        # Save outliers as a table
        outliers_df = df[dbscan_labels == -1]
        outliers_df.to_csv(os.path.join(assets_dir, 'dbscan_outliers.csv'), index=False)
        
        # 6. Analyze clusters (using K-means results)
        df_clustered, cluster_stats = analyze_clusters(df, kmeans_labels, feature_cols)
        
        # 7. Association rule mining
        medication_association_analysis()
        
        # 8. Network analysis
        network_analysis()
        
        # 9. Create dashboard
        create_cluster_dashboard(df_clustered, X_pca, kmeans_labels, assets_dir, max_samples=max_samples)
        
        # 10. Summary statistics
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total patients analyzed: {len(df):,}")
        print(f"Features used: {len(feature_cols)}")
        print(f"Optimal clusters (K-means): {optimal_k}")
        print(f"DBSCAN outliers: {np.sum(dbscan_labels == -1)}")
        
        print(f"\nCluster distribution:")
        cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = (count / len(kmeans_labels)) * 100
            print(f"  Cluster {cluster}: {count:,} patients ({percentage:.1f}%)")
        
        print(f"\nFiles saved in: {assets_dir}")
        print("- cluster_statistics.csv")
        print("- cluster_characteristics.png")
        print("- pca_analysis.png")
        print("- cluster_optimization.png")
        print("- medication_association_rules.csv")
        print("- patient_professional_network.png")
        print("- cluster_dashboard.html")
        print("- report.html")
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NHS Patient Segmentation Analysis")
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of patient samples to process')
    args = parser.parse_args()
    main(max_samples=args.max_samples)
