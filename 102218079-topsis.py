import pandas as pd
import numpy as np
data = pd.read_csv('102218079-data.csv')
funds = data['Fund Name']
matrix = data.drop('Fund Name', axis=1).values
weights = np.array([1, 1, 1, 1, 1]) 
impacts = np.array(['+', '+', '-', '+', '-'])  
norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
weighted_matrix = norm_matrix * weights
ideal_best= np.where(impacts == '+', np.max(weighted_matrix, axis=0), np.min(weighted_matrix, axis=0))
ideal_worst= np.where(impacts == '+', np.min(weighted_matrix, axis=0), np.max(weighted_matrix, axis=0))
distance_ideal_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
distance_ideal_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
topsis_score = distance_ideal_worst / (distance_ideal_best + distance_ideal_worst)
results = pd.DataFrame({
    'Fund Name': funds,
    'TOPSIS Score': topsis_score
})
results['Rank'] = results['TOPSIS Score'].rank(ascending=False).astype(int)
results = results.sort_values(by='Rank')
results.to_csv('102218079-output.csv',index=False)
print(results)