import pandas as pd
import numpy as np


# conditions form a sliding window
# condition_left_below: condition is true if data_frame 2 row is above window lower bound
# condition_left_above: condition is true if data_frame 2 row is below window upper bound

# TODO: check if assumption with sorted data is correct
def sortable_join(df1, df2, condition_left_below, condition_left_above, left_on, right_on, type='inner'):
    df1_sorted = df1.sort_values(by=left_on, kind='stable')
    df2_sorted = df2.sort_values(by=right_on, kind='stable')

    sorted_above_condition = lambda x, y: ((x[left_on].values == y[right_on].values).all()) and condition_left_above(x, y)
    sorted_below_condition = lambda x, y: ((x[left_on].values == y[right_on].values).all()) and condition_left_below(x, y)

    return two_pointer_join(df1_sorted, df2_sorted, sorted_below_condition, sorted_above_condition, right_on, left_on, type)


def two_pointer_join(df1, df2, condition_right_below, condition_right_above, right_on, left_on, type):
    j = 0
    result = []

    df1_join_partners = np.full(len(df1), True, dtype=bool)
    df2_join_partners = np.full(len(df2), False, dtype=bool)

    for i, left_row in df1.iterrows():

        while not condition_right_below(left_row, df2.iloc[j]):
            j += 1

        if not condition_right_above(left_row, df2.iloc[j]):
            continue

        df1_join_partners[i] = True
        test = df1.iloc[j]
        previous_j = j
        for k, right_row in df2.iloc[j:].iterrows():
            if not condition_right_above(left_row, right_row):
                break
            new_concat_rows(result, left_row, right_row, left_on, right_on)
            df2_join_partners[k] = True

        j = previous_j

    result_frame = pd.DataFrame(result).reset_index(drop=True)
    if type=='inner':
        return result_frame
    if type=='outer':
        right_joined = handle_right_join(df1, df2, right_on, df1_join_partners)
        left_joined = handle_left_join(df1, df2, left_on, right_on, df2_join_partners)
        return pd.concat([result_frame, right_joined, left_joined], ignore_index=True)
    if type=='left':
        left_joined = handle_left_join(df1, df2, left_on, right_on, df2_join_partners)
        return pd.concat([result_frame, left_joined], ignore_index=True)
    if type=='right':
        right_joined = handle_right_join(df1, df2, right_on, df1_join_partners)
        return pd.concat([result_frame, right_joined], ignore_index=True)

    return Exception('Invalid join type')

def handle_right_join(df1, df2, right_on, df1_join_partners):
    outer_rows = df1[df1_join_partners]
    new_columns = np.setdiff1d(df2.columns, np.union1d(df1.columns, right_on))
    outer_rows[new_columns] = np.nan
    return outer_rows

def handle_left_join(df1, df2, left_on, right_on, df2_join_partners):
    outer_rows = df2[df2_join_partners]
    rename_mapping = dict(zip(right_on, left_on))
    outer_rows = outer_rows.rename(columns=rename_mapping)
    new_columns = np.setdiff1d(df1.columns, outer_rows.columns)
    outer_rows[new_columns] = np.nan
    return outer_rows

def new_concat_rows(result, row1, row2, left_on, right_on):
    if (not len(left_on) > 0 and len(right_on) < 0) and [left_on] != row2[right_on]:
        return
    filtered_row2 = row2.drop(right_on)
    return result.append(pd.concat([row1, filtered_row2]))


def test(size):
    list = [i for i in range(size)]
    key = ["A", "A", "C", "C", "C", "B", "B", "B", "B", "B"]
    left = {
        'A': list,
        'B': list,
        'D': key
    }

    right = {
        'C': list,
        'E': key
    }

    left = pd.DataFrame(left)
    left['B'] = left['B'] + 2
    right = pd.DataFrame(right)

    condition_left_below = lambda x, y: x['A'] <= y['C']
    condition_left_above = lambda x, y: x['B'] >= y['C']

    sorted = sortable_join(left, right, condition_left_below, condition_left_above, ['D'], ['E'],'inner')
    print (sorted)

test(10)
