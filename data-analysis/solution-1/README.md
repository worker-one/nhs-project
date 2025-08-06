## Modelling Solution 1: Statistical Analysis of Treatment Outcomes and Healthcare Utilization

**Category:** Inferential Statistics

### Problem
The NHS needs to understand the statistical relationships between patient demographics, treatment patterns, and health outcomes to optimize resource allocation and improve patient care quality. Specifically, we need to determine if there are significant differences in treatment outcomes across different hospitals, departments, and patient demographics.

### Solution

Here we are trying to develop a comprehensive statistical analysis framework using:
- ANOVA tests to compare treatment outcomes across hospitals and departments
- Chi-square tests for categorical relationships (e.g., gender vs. treatment success)
- Multiple regression analysis to identify factors affecting treatment costs and duration
- Hypothesis testing for appointment wait times vs. patient satisfaction
- Confidence interval estimation for key performance metrics

### Justification
Proposed methods are well-established in the statistical science. They handle the complex, multi-table nature of healthcare data effectively.

- ANOVA is appropriate for comparing means across multiple groups (hospitals/departments)
- Regression analysis can identify key predictors of healthcare costs and outcomes
- Hypothesis testing provides evidence-based insights for policy decisions


### Implementation Technologies
We are using Python as a main tool for orchestration of tests. This language is historacly designed for scripting that it is way it is very popular in practical statistical science.

- Python with scipy.stats, statsmodels
- SQL queries joining Patients, Appointments, Surgeries, Tests, ServiceBillings, Hospitals, Departments
- Pandas for data manipulation and aggregation
- Matplotlib/Seaborn for statistical visualizations

### Expected Results
- Statistical significance of hospital performance differences
- Key demographic factors affecting treatment costs
- Evidence-based recommendations for resource allocation
- Performance benchmarks with confidence intervals

### Limitations
As any other method, the proposed method has its limitations.
- We have to assumes normal distribution for parametric tests
- Cannot establish causation, only correlation
- May require data transformation for non-normal distributions
- Sample size limitations for smaller hospitals/departments


### Results

We display all results in report `statistical_analysis_report.html` . Here are main insights:

#### REGRESSION ANALYSIS

This regression output provides insights into how different factors affect **treatment costs (Amount)** based on a linear model. Here's a step-by-step interpretation:

**Model Summary**

* **Dependent Variable**: `Amount` (treatment costs)
* **Observations**: 81,521
* **Features**: 6 predictors: `Age`, `Gender_F`, `Gender_M`, `Service_Appointment`, `Service_Surgery`, `Service_Test`
* **R-squared**: **0.746**

  * About **74.6%** of the variation in treatment costs is explained by the model—a **strong fit**.
* **Adjusted R-squared**: Also **0.746**, indicating good generalizability.
* **F-statistic**: Very high and significant (**p < 0.0001**), meaning **at least one predictor is significantly related to the cost**.

---

**Coefficients Interpretation**

Each coefficient represents the **estimated change in treatment cost** for a 1-unit increase in the variable, holding others constant:

| Variable                 | Coef    | P-value | Interpretation                                       |
| ------------------------ | ------- | ------- | ---------------------------------------------------- |
| **Intercept (const)**    | 555.72  | 0.000   | Baseline cost when all other vars = 0                |
| **Age**                  | -0.03   | 0.684   | Not significant (age does **not** affect cost)       |
| **Gender\_F**            | 279.06  | 0.000   | Being female adds \~\$279 to cost vs baseline        |
| **Gender\_M**            | 276.65  | 0.000   | Being male adds \~\$277 to cost vs baseline          |
| **Service\_Appointment** | -645.47 | 0.000   | Appointments cost \~\$645 **less** than the baseline |
| **Service\_Surgery**     | 1756.73 | 0.000   | Surgery costs \~\$1757 **more** than baseline        |
| **Service\_Test**        | -555.55 | 0.000   | Tests cost \~\$556 **less** than baseline            |

**Other Stats**

* **Durbin-Watson: 1.99** → Residuals are not autocorrelated (ideal value is \~2)
* **Omnibus, Jarque-Bera**: Large → Residuals **not normally distributed**, but with large samples, this is often tolerated.
* **Condition Number: 1.38e+17** → **Very high**, suggests **severe multicollinearity**—likely from including both gender dummies and all service types.

**Key Takeaways**

1. **Service type** is the strongest cost driver:

   * Surgery increases cost substantially.
   * Appointments and tests reduce cost vs the base category (possibly inpatient care or something omitted).
2. **Gender appears significant**, though the inclusion of both male and female dummies may distort this. Recode one as the reference.
3. **Age has no significant impact** on treatment cost.
4. Model explains a large portion of variance in cost (**R² = 0.746**).


#### CHI-SQUARE TESTS

Chi-square Test - Gender vs Surgery Outcome:
Chi-square statistic: 7.9468, p-value: 0.1592
No significant association is found.


#### ANOVA ANALYSIS
ANOVA - Surgery Outcomes by Hospital:F-statistic: 1.0490, p-value: 0.3286
No significant difference is found.