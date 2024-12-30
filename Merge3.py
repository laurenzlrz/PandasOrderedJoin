import pandas as pd
import numpy as np

# Todo: make join with condition clear
# Todo: Then make the filters clear
# TODO: check if assumption with sorted data is correct

# This class implements a Pandas Dataframe merge with condition operation using two-pointer join algorithm
# A requirement for this merge is a ordered condition
# Ordered Condition is defined as a condition that filters a "window" of possible matches:
# Take a arbitrary row i from the left dataframe. Then in the right dataframe, there are indizes j and k such that
# the condition is true for all rows in the right dataframe with indizes between j and k (Can also be none).
# For each other row i' > i in the left dataframe, it implies that j' >= j and k' >= k
# The complexity of this operation is O(n + m) where n is the number of rows in the left dataframe
# and m is the number of rows in the right dataframe and O(n log(n) + m log(m)) if key sorting has to be done

# The join on with condition works as follows:
# First the corresponding rows are matched based on the join keys, outer join keys are filled with NaN (if outer join)
# Afterwards the condition is checked for the matched rows
# If the condition is true, the rows are merged based on the keys and added to the result

# condition_left_below: condition true if row from left dataframe is low enough for a given row from right dataframe
# If its not true, it implies, that only, higher rows from the right dataframe can be a match
# or that lower rows from the left dataframe can be a match
# condition_right_below: condition true if row from right dataframe is low enough for a given row from left dataframe
# If its not true, it implies, that only, higher rows from the left dataframe can be a match
# or that lower rows from the right dataframe can be a match

# It is always assumed, that the dataframes are correctly sorted for the conditions, such that the formal assumption
# from above is met

# This function can be used, if the dataframes join keys are sortable.
# It still has to be ensured that the
def sortable_join(df1, df2, condition_left_below, condition_right_below, left_on, right_on, type='inner'):
    # Presorting is possible, since it is a stable sort and the order for the condition is not changed
    df1_sorted = df1.sort_values(by=left_on, kind='stable').reset_index(drop=True)
    df2_sorted = df2.sort_values(by=right_on, kind='stable').reset_index(drop=True)

    # Convert sorted DataFrames to NumPy structured arrays for efficient access
    df1_np = df1_sorted.to_records(index=False)
    df2_np = df2_sorted.to_records(index=False)

    # Call the two_pointer_join with the NumPy arrays and the condition functions
    return two_pointer_join(df1_np, df2_np, condition_left_below, condition_right_below, right_on, left_on, type)

# Implement if the join keys are sortable
def two_pointer_join(left_df, right_df, condition_left_below, condition_right_below, right_on, left_on, join_type):

    def primary_right_condition(x, y):
        # Extract field values directly without looping
        x_keys = np.array(x[left_on].tolist())
        y_keys = np.array(y[right_on].tolist())

        # Perform element-wise comparison
        keys_match = np.all(x_keys == y_keys)

        return keys_match and condition_right_below(x, y)

    def primary_left_condition(x, y):
        # Extract field values directly
        x_keys = np.array(x[left_on].tolist())
        y_keys = np.array(y[right_on].tolist())

        keys_match = np.all(x_keys == y_keys)

        return keys_match and condition_left_below(x, y)

    j = 0
    result = []

    df1_join_partners = np.full(len(left_df), False, dtype=bool)
    df2_join_partners = np.full(len(right_df), False, dtype=bool)

    for i in range(len(left_df)):
        left_row = left_df[i]

        while j < len(right_df) and not primary_left_condition(left_row, right_df[j]):
            j += 1

        if j >= len(right_df):
            break  # No more matches possible

        if not primary_right_condition(left_row, right_df[j]):
            continue

        # Mark the df1 row as having at least one join partner
        df1_join_partners[i] = True

        # Store the starting position to reset j later
        previous_j = j

        # Iterate over df2 starting from j
        for k in range(j, len(right_df)):
            right_row = right_df[k]

            # Check if the row satisfies the upper condition
            if not primary_right_condition(left_row, right_row):
                break  # Exit the inner loop as further rows won't satisfy the condition

            # Concatenate the rows based on the join keys
            concatenated = new_concat_rows_numpy(
                left_df, right_df, i, k, left_on, right_on
            )
            if concatenated is not None:
                result.append(concatenated)

            # Mark the df2 row as having a join partner
            df2_join_partners[k] = True

        # Reset j to the previous position for the next iteration
        j = previous_j

    # Convert the result list to a DataFrame
    if result:
        result_df = pd.DataFrame(result)
    else:
        # If no results, return an empty DataFrame with combined columns
        combined_columns = list(left_df.dtype.names) + [col for col in right_df.dtype.names if col not in right_on]
        result_df = pd.DataFrame(columns=combined_columns)

    # Handle different join types
    if join_type == 'inner':
        return result_df
    elif join_type == 'outer':
        right_joined = handle_right_join(left_df, right_df, right_on, df1_join_partners)
        left_joined = handle_left_join(left_df, right_df, left_on, right_on, df2_join_partners)
        return pd.concat([result_df, right_joined, left_joined], ignore_index=True)
    elif join_type == 'left':
        left_joined = handle_left_join(left_df, right_df, left_on, right_on, df2_join_partners)
        return pd.concat([result_df, left_joined], ignore_index=True)
    elif join_type == 'right':
        right_joined = handle_right_join(left_df, right_df, right_on, df1_join_partners)
        return pd.concat([result_df, right_joined], ignore_index=True)
    else:
        raise ValueError('Invalid join type')


def new_concat_rows_numpy(left_np_row, right_np_row, i, k, left_on, right_on):
    """
    Concatenates two rows from NumPy structured arrays into a single dictionary.
    Excludes the right_on columns from df2 to avoid duplication.
    """
    left_row = left_np_row[i]
    right_row = right_np_row[k]

    # Check if the join keys match
    for l_key, r_key in zip(left_on, right_on):
        if left_row[l_key] != right_row[r_key]:
            return None  # Join keys do not match; skip this pair

    # Create the combined row
    combined_row = {}
    for col in left_np_row.dtype.names:
        combined_row[col] = left_row[col]
    for col in right_np_row.dtype.names:
        if col not in right_on:
            combined_row[col] = right_row[col]
    return combined_row

# Add another filter
def handle_right_join(df1_np, df2_np, right_on, df1_join_partners):
    # Rows in df1 that have no join partners
    outer_rows = df1_np[~df1_join_partners]

    if len(outer_rows) == 0:
        return pd.DataFrame()

    # Convert to DataFrame
    outer_df = pd.DataFrame(outer_rows)

    # Add NaN for columns from df2
    new_columns = [col for col in df2_np.dtype.names if col not in right_on]
    for col in new_columns:
        outer_df[col] = np.nan
    return outer_df


def handle_left_join(df1_np, df2_np, left_on, right_on, df2_join_partners):
    # Rows in df2 that have no join partners
    outer_rows = df2_np[~df2_join_partners]
    if len(outer_rows) == 0:
        return pd.DataFrame()

    # Convert to DataFrame
    outer_df = pd.DataFrame(outer_rows)

    # Rename right_on columns to left_on names
    rename_mapping = {right: left for left, right in zip(left_on, right_on)}
    outer_df.rename(columns=rename_mapping, inplace=True)

    # Add NaN for columns from df1
    new_columns = [col for col in df1_np.dtype.names if col not in left_on]
    for col in new_columns:
        outer_df[col] = np.nan
    return outer_df


# Example usage
def test(size):
    lst = [i for i in range(size)]
    key = ["A", "A", "C", "C", "C", "B", "B", "B", "B", "B"]
    left = {
        'A': lst,
        'B': np.array(lst) + 2,  # Ensure 'B' is updated as per original code
        'D': key
    }

    right = {
        'C': lst,
        'E': key
    }

    left_df = pd.DataFrame(left)
    right_df = pd.DataFrame(right)

    # Example: Both conditions create a window of possible matches
    # Entries in the left dataframe are bound by the right dataframe and vice versa
    condition_left_below = lambda x, y: x['A'] <= y['C']
    condition_right_below = lambda x, y: x['B'] >= y['C']

    sorted_df = sortable_join(left_df, right_df, condition_left_below, condition_right_below, ['D'], ['E'], 'inner')
    print(sorted_df)


if __name__ == "__main__":
    test(10)
