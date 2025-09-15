import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

data = pd.read_csv('gene_expression_data.csv', index_col=0)

sample_groups = {
    'sample1': 'control',
    'sample2': 'control',
    'sample3': 'treatment',
    'sample4': 'treatment'
}

control_samples = [sample for sample in data.columns if sample_groups[sample] == 'control']
treatment_samples = [sample for sample in data.columns if sample_groups[sample] == 'treatment']

results = []
for gene in data.index:
    control_vals = data.loc[gene, control_samples]
    treatment_vals = data.loc[gene, treatment_samples]

    t_stat, p_val = ttest_ind(control_vals, treatment_vals, equal_var=False)

    mean_diff = treatment_vals.mean() - control_vals.mean()

    results.append({
        'Gene': gene,
        'MeanDifference': mean_diff,
        'P-value': p_val
    })

results_df = pd.DataFrame(results)

from statsmodels.stats.multitest import multipletests

results_df['Adj_P-value'] = multipletests(results_df['P-value'], method='fdr_bh')[1]

results_df = results_df.sort_values('Adj_P-value')

print(results_df.head(10))
