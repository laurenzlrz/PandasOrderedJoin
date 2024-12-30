import numpy as np
import pandas as pd

def keys_sorted_ordered_join(left_df, right_df, condition_left_below, condition_right_below, right_on, left_on):

    def process_inner_rows(left_row_number, right_inner_rows, left_inner_rows, new_start):
        right_row_number = 0
        while right_row_number < len(right_df):
            right_row_number += 1
            if not join_below_condition(left_df[left_row_number], right_df[right_row_number]):
                new_start = right_row_number
                continue
            if not join_above_condition(left_df[left_row_number], right_df[right_row_number]):
                break
            right_inner_rows[right_row_number] = True
            left_inner_rows[left_row_number] = True
            if not condition_left_below(left_df[left_row_number], right_df[right_row_number]):
                new_start = right_row_number
                continue
            if not condition_right_below(left_df[left_row_number], right_df[right_row_number]):
                break
        return new_start

    def process_outer_rows(left_row_number, right_inner_rows, left_inner_rows, last_outer_checked):
        for right_row_number in range(last_outer_checked, len(right_df)):
            last_outer_checked += 1
            if not join_below_condition(left_df[left_row_number], right_df[right_row_number]):
                continue
            if not join_above_condition(left_df[left_row_number], right_df[right_row_number]):
                break
            right_inner_rows[right_row_number] = True
            left_inner_rows[left_row_number] = True
        return last_outer_checked

    left_inner_rows = np.full(len(left_df), False, dtype=bool)
    right_inner_rows = np.full(len(right_df), False, dtype=bool)

    new_start = 0
    last_outer_checked = 0
    for left_row_number in range(len(left_df)):
        new_start = process_inner_rows(left_row_number, right_inner_rows, left_inner_rows, new_start)
        last_outer_checked = process_outer_rows(left_row_number, right_inner_rows, left_inner_rows, last_outer_checked)

    return new_start

def iterate_until_not_with_keys(right_df, left_row, i, j, condition_join_below, condition_join_above, right_inner_rows, left_inner_rows):
    new_start = j
    row_number = j
    while row_number < range(j, len(right_df)):
        row_number += 1
        if not condition_join_below(left_row, right_df[row_number]):
            new_start += 1
            continue
        if not condition_join_above(left_row, right_df[row_number]):
            break

        right_inner_rows[row_number] = True
        left_inner_rows[i] = True

        if not condition_left_below(left_row, right_df[row_number]):
            new_start += 1
            continue

        if not condition_right_below(left_row, right_df[row_number]):
            break
         #addRow

    return new_start

def iterate_outer(right_df, left_row, i, last_join_checked, condition_join_below, condition_join_above, right_inner_rows, left_inner_rows):

    for row_number in range(last_join_checked, len(right_df)):
        last_join_checked += 1
        if not condition_join_below(left_row, right_df[row_number]):
            continue
        if not condition_join_above(left_row, right_df[row_number]):
            break

        right_inner_rows[row_number] = True
        left_inner_rows[i] = True

    return last_join_checked

def iterate_until_not(right_df, left_row, i, j, condition_left_below, condition_right_below):
    k = j
    for right_row in right_df[j,:]:
        if not condition_left_below(left_row, right_row):
            k += 1
            continue
        if not condition_right_below(left_row, right_row):
            break

        # addRow
    return k

def join(left_df, right_df):
    j = 0
    result = []

    for i in range(len(left_df)):
        left_row = left_df[i]

        # Means did not find any row to join on. Belongs to outer join, if not previously joined.
        # No need to differentiata because join partners set in later loop.
        while j < len(right_df) and condition_left_below(left_row, right_df[j]):
            j += 1

        if j >= len(right_df):
            break  # No more matches possible

        # Store the starting position to reset j later
        previous_j = j

        # Iterate over df2 starting from j
        for k in range(j, len(right_df)):
            right_row = right_df[k]

            # Means we failed to join because of the condition
            if not condition_right_below(left_row, right_row):
                break

            if not joinon_condition(left_row, right_row):
                continue

            # Concatenate the rows based on the join keys
            concatenated = new_concat_rows_numpy(
                left_df, right_df, i, k, left_on, right_on
            )
            if concatenated is not None:
                result.append(concatenated)

            # Mark the df2 row as having a join partner

        # Reset j to the previous position for the next iteration
        j = previous_j

    return pd.concat([pd.DataFrame(result),
                      check_if_outer_applies(left_df, right_df, condition_left_below,
                                                                   condition_right_below,
                                                                   left_on, right_on)], ignore_index=True)


# This function checks for outer and inner joins, when not ordered
def check_if_outer_applies(left_df, right_df, condition_left_below, condition_right_below, left_on, right_on):

    right_result = check_right_columns(left_df, right_df, condition_left_below, condition_right_below, left_on, right_on)
    left_result = check_left_columns(left_df, right_df, condition_left_below, condition_right_below, right_on)

    outer_right_result_df = right_result[~right_result[left_on].isin(left_df[left_on])]
    outer_left_result_df = left_result[~left_result[right_on].isin(right_result[left_on])]

    return pd.concat([outer_left_result_df, outer_right_result_df], ignore_index=True)


def check_right_columns(left_df, right_df, condition_left_below, condition_right_below, left_on, right_on):
    result = []
    left_nan_row = pd.Series(data=np.nan, index=left_df.columns)
    for row in right_df.iterrows():
        if condition_left_below(left_nan_row, row) and condition_right_below(left_nan_row, row):
            result.append(pd.concat([left_nan_row, row]))
    result_df = pd.DataFrame(result).drop(columns=left_on)
    rename_dict = dict(zip(right_on, left_on))
    return result_df.rename(columns=rename_dict)

def check_left_columns(left_df, right_df, condition_left_below, condition_right_below, right_on):
    result = []
    right_nan_row = pd.Series(data=np.nan, index=right_df.columns)
    for row in left_df.iterrows():
        if condition_left_below(row, right_nan_row) and condition_right_below(row, right_nan_row):
            result.append(row)
    return pd.DataFrame(result).drop(columns=right_on)



# def keys_sorted_ordered_join(left_df, right_df, condition_left_below, condition_right_below, right_on, left_on):
#    def join_below_condition(x, y):
#        # Extract field values directly without looping
#        x_keys = np.array(x[left_on].tolist())
#        y_keys = np.array(y[right_on].tolist())
#        return np.all(x_keys <= y_keys)
#
#    def join_above_condition(x, y):
#        # Extract field values directly
#        x_keys = np.array(x[left_on].tolist())
#        y_keys = np.array(y[right_on].tolist())
#        return np.all(x_keys >= y_keys)
#
#    left_inner_rows = np.full(len(left_df), False, dtype=bool)
#    right_inner_rows = np.full(len(right_df), False, dtype=bool)
#
#    new_start = 0
#    last_outer_checked = 0
#    for left_row_number in range(len(left_df)):
#
#        right_row_number = 0
#        while right_row_number < range(j, len(right_df)):
#            right_row_number += 1
#
#            if not join_below_condition(left_df[left_row_number], right_df[right_row_number]):
#                new_start = right_row_number
#                continue
#            if not join_above_condition(left_df[left_row_number], right_df[right_row_number]):
#                break
#
#            right_inner_rows[right_row_number] = True
#            left_inner_rows[left_row_number] = True
#
#            if not condition_left_below(left_df[left_row_number], right_df[right_row_number]):
#                new_start = right_row_number
#                continue
#
#            if not condition_right_below(left_df[left_row_number], right_df[right_row_number]):
#                break
#             #addRow
#
#        for right_row_number in range(last_outer_checked, len(right_df)):
#            last_outer_checked += 1
#            if not join_below_condition(left_df[left_row_number], right_df[right_row_number]):
#                continue
#            if not join_above_condition(left_df[left_row_number], right_df[right_row_number]):
#                break
#
#            right_inner_rows[right_row_number] = True
#            left_inner_rows[left_row_number] = True
#
#        return last_outer_checked
#
#    return new_start