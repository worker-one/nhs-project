import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import sessionmaker
import os
import warnings
import base64
warnings.filterwarnings('ignore')

# Import database and models
import sys
sys.path.append('..')
from database import engine
sys.path.append('../../sqlite-scripts')
from models import *

# Create session
Session = sessionmaker(bind=engine)
session = Session()

# Create assets directory
assets_dir = './assets'
os.makedirs(assets_dir, exist_ok=True)

def load_data():
    """Load and prepare data for analysis"""
    
    # Load surgeries with hospital and patient info
    surgery_query = """
    SELECT s.SurgeryID, s.PatientID, s.ProfessionalID, s.HospitalID, s.Date, 
           s.Type, s.Outcome, p.Gender, p.DOB,
           h.Name as HospitalName, h.Location as HospitalLocation,
           CASE 
               WHEN s.Outcome = 'Successful' THEN 1
               WHEN s.Outcome = 'Partial Success' THEN 0.5
               ELSE 0
           END as OutcomeScore
    FROM Surgeries s
    JOIN Patients p ON s.PatientID = p.PatientID
    JOIN Hospitals h ON s.HospitalID = h.HospitalID
    """
    
    # Load appointments with department and wait time info
    appointment_query = """
    SELECT a.AppointmentID, a.PatientID, a.ProfessionalID, a.DepartmentID,
           a.DateTime, a.Status, p.Gender, p.DOB,
           d.Name as DepartmentName, h.Name as HospitalName,
           julianday(a.DateTime) - julianday('2024-01-01') as DaysFromStart
    FROM Appointments a
    JOIN Patients p ON a.PatientID = p.PatientID
    JOIN Departments d ON a.DepartmentID = d.DepartmentID
    JOIN Hospitals h ON d.HospitalID = h.HospitalID
    """
    
    # Load billing data
    billing_query = """
    SELECT sb.BillingID, sb.PatientID, sb.Amount, sb.PaymentStatus,
           sb.AmountPaid, p.Gender, p.DOB,
           CASE 
               WHEN sb.AppointmentID IS NOT NULL THEN 'Appointment'
               WHEN sb.SurgeryID IS NOT NULL THEN 'Surgery'
               WHEN sb.TestID IS NOT NULL THEN 'Test'
               ELSE 'Other'
           END as ServiceType
    FROM ServiceBillings sb
    JOIN Patients p ON sb.PatientID = p.PatientID
    """
    
    surgeries_df = pd.read_sql(surgery_query, engine)
    appointments_df = pd.read_sql(appointment_query, engine)
    billing_df = pd.read_sql(billing_query, engine)
    
    return surgeries_df, appointments_df, billing_df

def perform_anova_analysis(surgeries_df):
    """Perform ANOVA tests comparing treatment outcomes across hospitals and departments"""

    print("=== ANOVA ANALYSIS ===")

    # Compute mean outcome score per hospital
    hospital_stats = surgeries_df.groupby('HospitalName')['OutcomeScore'].agg(['mean', 'std', 'count'])
    hospital_stats = hospital_stats.sort_values('mean', ascending=False)

    # Select top 10 and bottom 10 hospitals by mean outcome score (with at least 100 surgeries)
    eligible_hospitals = hospital_stats[hospital_stats['count'] > 100]
    top10 = eligible_hospitals.head(10).index.tolist()
    bottom10 = eligible_hospitals.tail(10).index.tolist()

    # Prepare data for ANOVA (all eligible hospitals)
    hospitals = eligible_hospitals.index.tolist()
    hospital_outcomes = [surgeries_df[surgeries_df['HospitalName'] == h]['OutcomeScore'].dropna()
                         for h in hospitals]

    if len(hospital_outcomes) > 1:
        f_stat, p_value = f_oneway(*hospital_outcomes)
        print(f"ANOVA - Surgery Outcomes by Hospital:")
        print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

        # Plot: Top 10 hospitals
        plt.figure(figsize=(12, 6))
        top10_df = surgeries_df[surgeries_df['HospitalName'].isin(top10)]
        top10_df.boxplot(column='OutcomeScore', by='HospitalName', ax=plt.gca())
        plt.title('Top 10 Hospitals by Mean Surgery Outcome Score')
        plt.suptitle('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{assets_dir}/anova_top10_surgery_outcomes_by_hospital.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot: Bottom 10 hospitals
        plt.figure(figsize=(12, 6))
        bottom10_df = surgeries_df[surgeries_df['HospitalName'].isin(bottom10)]
        bottom10_df.boxplot(column='OutcomeScore', by='HospitalName', ax=plt.gca())
        plt.title('Bottom 10 Hospitals by Mean Surgery Outcome Score')
        plt.suptitle('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{assets_dir}/anova_bottom10_surgery_outcomes_by_hospital.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("\nHospital Surgery Outcome Statistics:")
        print(hospital_stats)

    return f_stat, p_value

def perform_chi_square_tests(surgeries_df, appointments_df):
    """Perform Chi-square tests for categorical relationships"""
    
    print("\n=== CHI-SQUARE TESTS ===")
    
    # Chi-square: Gender vs Surgery Outcome
    gender_outcome_crosstab = pd.crosstab(surgeries_df['Gender'], surgeries_df['Outcome'])
    chi2, p_value, dof, expected = chi2_contingency(gender_outcome_crosstab)
    
    print(f"Chi-square Test - Gender vs Surgery Outcome:")
    print(f"Chi-square statistic: {chi2:.4f}, p-value: {p_value:.4f}")
    print(f"Significant association: {'Yes' if p_value < 0.05 else 'No'}")
    print("\nCrosstab - Gender vs Surgery Outcome:")
    print(gender_outcome_crosstab)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    gender_outcome_crosstab.plot(kind='bar', ax=plt.gca())
    plt.title('Surgery Outcomes by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Outcome')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{assets_dir}/chi_square_gender_vs_outcome.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chi-square: Gender vs Appointment Status
    if 'Status' in appointments_df.columns:
        gender_status_crosstab = pd.crosstab(appointments_df['Gender'], appointments_df['Status'])
        chi2_2, p_value_2, dof_2, expected_2 = chi2_contingency(gender_status_crosstab)
        
        print(f"\nChi-square Test - Gender vs Appointment Status:")
        print(f"Chi-square statistic: {chi2_2:.4f}, p-value: {p_value_2:.4f}")
        print(f"Significant association: {'Yes' if p_value_2 < 0.05 else 'No'}")
    
    return chi2, p_value

def perform_regression_analysis(billing_df, surgeries_df):
    """Perform multiple regression analysis for treatment costs"""
    
    print("\n=== REGRESSION ANALYSIS ===")
    
    # Check if we have enough data
    if len(billing_df) < 10:
        print("Insufficient data for regression analysis")
        return None
    
    # Prepare data for regression
    billing_df_copy = billing_df.copy()
    
    # Convert DOB to age, handle missing values
    try:
        billing_df_copy['DOB'] = pd.to_datetime(billing_df_copy['DOB'], errors='coerce')
        billing_df_copy['Age'] = 2024 - billing_df_copy['DOB'].dt.year
    except:
        billing_df_copy['Age'] = 50  # Default age
    
    # Clean the data - remove rows with missing critical values
    billing_df_copy = billing_df_copy.dropna(subset=['Amount'])
    billing_df_copy = billing_df_copy[billing_df_copy['Amount'] > 0]
    
    # Fill missing ages with median
    if 'Age' in billing_df_copy.columns:
        median_age = billing_df_copy['Age'].median()
        billing_df_copy['Age'] = billing_df_copy['Age'].fillna(median_age)
    else:
        billing_df_copy['Age'] = 50  # Default age
    
    # Ensure Age is numeric
    billing_df_copy['Age'] = pd.to_numeric(billing_df_copy['Age'], errors='coerce').fillna(50)
    
    # Handle categorical variables - fill missing values first
    billing_df_copy['Gender'] = billing_df_copy['Gender'].fillna('Unknown')
    billing_df_copy['ServiceType'] = billing_df_copy['ServiceType'].fillna('Other')
    
    # Create dummy variables for categorical variables
    try:
        billing_df_encoded = pd.get_dummies(billing_df_copy, columns=['Gender', 'ServiceType'], prefix=['Gender', 'Service'])
    except Exception as e:
        print(f"Error creating dummy variables: {e}")
        return None
    
    # Select features for regression - ensure they exist and are numeric
    feature_cols = []
    if 'Age' in billing_df_encoded.columns:
        feature_cols.append('Age')
    dummy_cols = [col for col in billing_df_encoded.columns if col.startswith(('Gender_', 'Service_'))]
    feature_cols.extend(dummy_cols)
    if len(feature_cols) == 0:
        print("No suitable features found for regression analysis")
        return None
    
    # Prepare final dataset
    X = billing_df_encoded[feature_cols].copy()
    y = billing_df_encoded['Amount'].copy()
    
    # Ensure all features are numeric and of float64 type
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(np.float64)
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(np.float64)
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) < 5:
        print("Insufficient clean data for regression analysis")
        return None
    
    print(f"Regression analysis using {len(X)} observations with {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    
    # Add constant term
    X = sm.add_constant(X)
    
    # Fit regression model
    try:
        model = sm.OLS(y, X).fit()
        
        print("Multiple Regression Results - Treatment Costs:")
        print(model.summary())
        
        # Create residual plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Fitted
        ax1.scatter(model.fittedvalues, model.resid, alpha=0.6)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted Values')
        
        # Q-Q plot
        stats.probplot(model.resid, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.savefig(f'{assets_dir}/regression_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return model
        
    except Exception as e:
        print(f"Error fitting regression model: {e}")
        print("This may be due to insufficient data variation or multicollinearity")
        return None

def calculate_confidence_intervals(surgeries_df, billing_df):
    """Calculate confidence intervals for key performance metrics"""
    
    print("\n=== CONFIDENCE INTERVALS ===")
    
    # CI for mean surgery outcome score
    outcome_scores = surgeries_df['OutcomeScore'].dropna()
    outcome_mean = outcome_scores.mean()
    outcome_se = stats.sem(outcome_scores)
    outcome_ci = stats.t.interval(0.95, len(outcome_scores)-1, loc=outcome_mean, scale=outcome_se)
    
    print(f"Surgery Outcome Score (0-1 scale):")
    print(f"Mean: {outcome_mean:.4f}")
    print(f"95% Confidence Interval: ({outcome_ci[0]:.4f}, {outcome_ci[1]:.4f})")
    
    # CI for mean treatment cost
    treatment_costs = billing_df['Amount'].dropna()
    cost_mean = treatment_costs.mean()
    cost_se = stats.sem(treatment_costs)
    cost_ci = stats.t.interval(0.95, len(treatment_costs)-1, loc=cost_mean, scale=cost_se)
    
    print(f"\nTreatment Costs:")
    print(f"Mean: £{cost_mean:.2f}")
    print(f"95% Confidence Interval: £({cost_ci[0]:.2f}, {cost_ci[1]:.2f})")
    
    # CI for proportion of successful surgeries
    successful_surgeries = (surgeries_df['Outcome'] == 'Successful').sum()
    total_surgeries = len(surgeries_df)
    success_rate = successful_surgeries / total_surgeries
    success_se = np.sqrt(success_rate * (1 - success_rate) / total_surgeries)
    success_ci = stats.norm.interval(0.95, loc=success_rate, scale=success_se)
    
    print(f"\nSurgery Success Rate:")
    print(f"Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
    print(f"95% Confidence Interval: ({success_ci[0]:.4f}, {success_ci[1]:.4f})")
    
    return {
        'outcome_ci': outcome_ci,
        'cost_ci': cost_ci,
        'success_rate_ci': success_ci
    }

def generate_summary_report(results):
    """Generate HTML summary report with embedded plots"""

    # Helper to embed image as base64
    def embed_image_base64(filepath):
        try:
            with open(filepath, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                return f'<img src="data:image/png;base64,{encoded}" style="max-width:700px; margin:20px 0; border:1px solid #ccc; border-radius:4px;" />'
        except Exception as e:
            return f'<p style="color:red;">Plot not available: {os.path.basename(filepath)}</p>'

    # Embed plots if available
    anova_top10_plot = embed_image_base64(f'{assets_dir}/anova_top10_surgery_outcomes_by_hospital.png')
    anova_bottom10_plot = embed_image_base64(f'{assets_dir}/anova_bottom10_surgery_outcomes_by_hospital.png')
    chi_plot = embed_image_base64(f'{assets_dir}/chi_square_gender_vs_outcome.png')
    # reg_plot = embed_image_base64(f'{assets_dir}/regression_diagnostics.png')  # No longer used

    # Get OLS regression summary table as HTML if available
    regression_html = results.get('regression_html', '<p>No regression results available.</p>')

    # Add regression meta info if available
    regression_meta = ""
    if results.get('regression_meta'):
        meta = results['regression_meta']
        regression_meta = f"""
        <table style="margin-bottom:20px;">
            <tr><th>Dep. Variable</th><td>{meta.get('dep_var','')}</td></tr>
            <tr><th>R-squared (adj.)</th><td>{meta.get('rsq_adj','')}</td></tr>
            <tr><th>No. Observations</th><td>{meta.get('nobs','')}</td></tr>
        </table>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NHS Statistical Analysis Report - Solution 1</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #005EB8; }}
            .metric {{ background-color: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .significant {{ color: #d63384; font-weight: bold; }}
            .not-significant {{ color: #6c757d; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #005EB8; color: white; }}
            .plot-section {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>NHS Data Intelligence - Statistical Analysis Report</h1>
        <h2>Solution 1: Treatment Outcomes and Healthcare Utilization Analysis</h2>
        
        <div class="metric">
            <h3>Key Findings Summary</h3>
            <ul>
                <li>ANOVA analysis reveals {'significant' if results.get('anova_p', 1) < 0.05 else 'no significant'} differences in surgery outcomes between hospitals</li>
                <li>Chi-square tests show {'significant' if results.get('chi2_p', 1) < 0.05 else 'no significant'} association between gender and surgery outcomes</li>
                <li>Multiple regression analysis identifies key cost predictors in healthcare services</li>
                <li>Confidence intervals provide reliable estimates for performance benchmarking</li>
            </ul>
        </div>
        
        <h3>Statistical Test Results</h3>
        <table>
            <tr><th>Test</th><th>Statistic</th><th>P-value</th><th>Significance</th></tr>
            <tr>
                <td>ANOVA - Surgery Outcomes by Hospital</td>
                <td>{results.get('anova_f', 'N/A'):.4f}</td>
                <td>{results.get('anova_p', 'N/A'):.4f}</td>
                <td class="{'significant' if results.get('anova_p', 1) < 0.05 else 'not-significant'}">
                    {'Significant' if results.get('anova_p', 1) < 0.05 else 'Not Significant'}
                </td>
            </tr>
            <tr>
                <td>Chi-square - Gender vs Surgery Outcome</td>
                <td>{results.get('chi2_stat', 'N/A'):.4f}</td>
                <td>{results.get('chi2_p', 'N/A'):.4f}</td>
                <td class="{'significant' if results.get('chi2_p', 1) < 0.05 else 'not-significant'}">
                    {'Significant' if results.get('chi2_p', 1) < 0.05 else 'Not Significant'}
                </td>
            </tr>
        </table>
        
        <h3>Performance Metrics with 95% Confidence Intervals</h3>
        <div class="metric">
            <p><strong>Surgery Success Rate:</strong> {results.get('success_rate', 0)*100:.2f}% 
            (CI: {results.get('success_ci', [0,0])[0]*100:.2f}% - {results.get('success_ci', [0,0])[1]*100:.2f}%)</p>
            <p><strong>Average Treatment Cost:</strong> £{results.get('cost_mean', 0):.2f} 
            (CI: £{results.get('cost_ci', [0,0])[0]:.2f} - £{results.get('cost_ci', [0,0])[1]:.2f})</p>
        </div>

        <div class="plot-section">
            <h3>Visualizations</h3>
            <h4>Top 10 Hospitals by Mean Surgery Outcome Score</h4>
            {anova_top10_plot}
            <h4>Bottom 10 Hospitals by Mean Surgery Outcome Score</h4>
            {anova_bottom10_plot}
            <h4>Chi-square: Surgery Outcomes by Gender</h4>
            {chi_plot}
            <h4>OLS Regression Results</h4>
            {regression_meta}
            {regression_html}
        </div>
        
        <h3>Recommendations</h3>
        <ul>
            <li>Focus quality improvement efforts on hospitals with below-average outcome scores</li>
            <li>Investigate gender-based treatment differences if statistically significant</li>
            <li>Use regression model coefficients to optimize resource allocation</li>
            <li>Implement performance monitoring using established confidence intervals</li>
        </ul>
        
        <p><em>Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </body>
    </html>
    """

    with open(f'{assets_dir}/statistical_analysis_report.html', 'w') as f:
        f.write(html_content)

    print(f"\nSummary report saved to {assets_dir}/statistical_analysis_report.html")

def main():
    """Main analysis function"""
    
    print("NHS Statistical Analysis - Solution 1")
    print("=====================================")
    
    # Load data
    print("Loading data...")
    surgeries_df, appointments_df, billing_df = load_data()
    
    print(f"Loaded {len(surgeries_df)} surgeries, {len(appointments_df)} appointments, {len(billing_df)} billing records")
    
    # Perform analyses
    results = {}
    
    # ANOVA Analysis
    try:
        f_stat, p_value = perform_anova_analysis(surgeries_df)
        results['anova_f'] = f_stat
        results['anova_p'] = p_value
    except Exception as e:
        print(f"Error in ANOVA analysis: {e}")
        results['anova_f'] = 'N/A'
        results['anova_p'] = 1.0
    
    # Chi-square Tests
    try:
        chi2_stat, chi2_p = perform_chi_square_tests(surgeries_df, appointments_df)
        results['chi2_stat'] = chi2_stat
        results['chi2_p'] = chi2_p
    except Exception as e:
        print(f"Error in Chi-square tests: {e}")
        results['chi2_stat'] = 'N/A'
        results['chi2_p'] = 1.0
    
    # Regression Analysis
    try:
        model = perform_regression_analysis(billing_df, surgeries_df)
        if model is not None:
            results['regression_r2'] = model.rsquared
            # Save OLS summary as HTML table for report
            try:
                results['regression_html'] = model.summary().tables[1].as_html()
            except Exception:
                # Fallback: full summary as HTML if possible
                try:
                    results['regression_html'] = model.summary().as_html()
                except Exception:
                    results['regression_html'] = '<p>Regression summary not available.</p>'
            # Extract meta info for display
            try:
                summ = model.summary()
                results['regression_meta'] = {
                    'dep_var': getattr(summ, 'dep_var', 'Amount'),
                    'rsq_adj': f"{getattr(model, 'rsquared_adj', 'N/A'):.4f}",
                    'nobs': f"{int(getattr(model, 'nobs', 0))}"
                }
            except Exception:
                results['regression_meta'] = {}
        else:
            results['regression_r2'] = 'N/A'
            results['regression_html'] = '<p>No regression results available.</p>'
            results['regression_meta'] = {}
    except Exception as e:
        print(f"Error in regression analysis: {e}")
        results['regression_r2'] = 'N/A'
        results['regression_html'] = '<p>No regression results available.</p>'
        results['regression_meta'] = {}
    
    # Confidence Intervals
    try:
        ci_results = calculate_confidence_intervals(surgeries_df, billing_df)
        results.update(ci_results)
    except Exception as e:
        print(f"Error calculating confidence intervals: {e}")
        # Set default values
        results['outcome_ci'] = (0, 1)
        results['cost_ci'] = (0, 1000)
        results['success_rate_ci'] = (0, 1)
    
    # Calculate additional metrics for report
    try:
        results['success_rate'] = (surgeries_df['Outcome'] == 'Successful').mean() if len(surgeries_df) > 0 else 0
        results['cost_mean'] = billing_df['Amount'].mean() if len(billing_df) > 0 else 0
        results['success_ci'] = results.get('success_rate_ci', (0, 1))
        results['cost_ci'] = results.get('cost_ci', (0, 1000))
    except Exception as e:
        print(f"Error calculating summary metrics: {e}")
        results['success_rate'] = 0
        results['cost_mean'] = 0
        results['success_ci'] = (0, 1)
        results['cost_ci'] = (0, 1000)
    
    # Generate summary report
    generate_summary_report(results)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Results and visualizations saved to {assets_dir}/")
    print("Key findings:")
    print(f"- Hospital outcome differences: {'Significant' if results.get('anova_p', 1) < 0.05 else 'Not significant'} (p={results.get('anova_p', 'N/A')})")
    print(f"- Gender-outcome association: {'Significant' if results.get('chi2_p', 1) < 0.05 else 'Not significant'} (p={results.get('chi2_p', 'N/A')})")
    print(f"- Surgery success rate: {results['success_rate']*100:.2f}%")
    print(f"- Average treatment cost: £{results['cost_mean']:.2f}")

if __name__ == "__main__":
    main()
