"""
SubmissionGenerator.py
Generates a submission file by combining results from several CSVs.
"""

import pandas as pd

def main(a_star_csv, breadth_first_csv):
    df_a_star = None
    df_breadth_first = None
    with open(a_star_csv, 'r') as fp:
        df_a_star = pd.read_csv(fp)
    with open(breadth_first_csv, 'r') as fp:
        df_breadth_first = pd.read_csv(fp)
    df_solutions = pd.DataFrame(columns=['Id','breadth_first_search','a_star_search'])
    solutions = []
    num_puzzles = 103
    for i in range(103):
        breadth_first_sol = None
        a_star_sol = None
        sol_series_bf = df_breadth_first.loc[(df_breadth_first.puzzle_id == i)]
        sol_series_as = df_a_star.loc[(df_a_star.puzzle_id == i)]
        if len(sol_series_bf) > 0:
            breadth_first_sol = df_breadth_first.iloc[sol_series_bf.index[0]]['solution']
        if len(sol_series_as) > 0:
            a_star_sol = df_a_star.iloc[sol_series_as.index[0]]['solution']
        solutions.append({'Id': i, 'breadth_first_search': breadth_first_sol, 'a_star_search': a_star_sol})
    df_solutions = pd.DataFrame.from_records(solutions, index='Id')
    with open('../prog2/submissions/test_submissions.csv', 'w') as fp:
        df_solutions.to_csv(fp)

if __name__ == '__main__':
    a_star_csv = '../prog2/solutions/a_star_solutions.csv'
    breadth_first_csv = '../prog2/solutions/breadth_first_solutions.csv'
    main(a_star_csv=a_star_csv, breadth_first_csv=breadth_first_csv)
