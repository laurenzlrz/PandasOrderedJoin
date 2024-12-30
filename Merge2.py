import pandas as pd
import numpy as np

# conditions form a sliding window
# condition_left_below: condition is true if data_frame 2 row is above window lower bound
# condition_left_above: condition is true if data_frame 2 row is below window upper bound

def sortable_join(df1, df2, condition_left_below, condition_left_above, left_on, right_on, type='inner'):
    """
    Sorts the input DataFrames and performs a two-pointer join based on the provided conditions.

    Parameters:
    - df1, df2: Input pandas DataFrames.
    - condition_left_below, condition_left_above: Functions defining the join conditions.
    - left_on, right_on: Lists of column names to join on.
    - type: Type of join ('inner', 'outer', 'left', 'right').

    Returns:
    - A pandas DataFrame resulting from the join operation.
    """
    df1_sorted = df1.sort_values(by=left_on, kind='stable').reset_index(drop=True)
    df2_sorted = df2.sort_values(by=right_on, kind='stable').reset_index(drop=True)

    sorted_above_condition = lambda x, y: ((x[left_on].values == y[right_on].values).all()) and condition_left_above(x, y)
    sorted_below_condition = lambda x, y: ((x[left_on].values == y[right_on].values).all()) and condition_left_below(x, y)

    return two_pointer_join(df1_sorted, df2_sorted, sorted_below_condition, sorted_above_condition, right_on, left_on, type)


def two_pointer_join(df1, df2, condition_right_below, condition_right_above, right_on, left_on, join_type):
    """
    Performs a two-pointer join on sorted DataFrames using NumPy structured arrays for efficiency.

    Parameters:
    - df1, df2: Sorted pandas DataFrames.
    - condition_right_below, condition_right_above: Functions defining the join conditions.
    - right_on, left_on: Lists of column names to join on.
    - join_type: Type of join ('inner', 'outer', 'left', 'right').

    Returns:
    - A pandas DataFrame resulting from the join operation.
    """
    # Convert DataFrames to NumPy structured arrays
    df1_np = df1.to_records(index=False)
    df2_np = df2.to_records(index=False)

    # Initialize pointers and result storage
    j = 0
    result = []

    df1_join_partners = np.full(len(df1_np), False, dtype=bool)
    df2_join_partners = np.full(len(df2_np), False, dtype=bool)

    for i in range(len(df1_np)):
        left_row = df1_np[i]

        # Advance the pointer j until condition_right_below is met
        while j < len(df2_np) and not condition_right_below(left_row, df2_np[j]):
            j += 1

        if j >= len(df2_np):
            break  # No more matches possible

        # Check if the current j satisfies condition_right_above
        if not condition_right_above(left_row, df2_np[j]):
            continue

        # Mark the df1 row as having at least one join partner
        df1_join_partners[i] = True

        # Store the starting position to reset j later
        previous_j = j

        # Iterate over df2 starting from j
        for k in range(j, len(df2_np)):
            right_row = df2_np[k]

            # Check if the row satisfies the upper condition
            if not condition_right_above(left_row, right_row):
                break  # Exit the inner loop as further rows won't satisfy the condition

            # Concatenate the rows based on the join keys
            # Exclude right_on columns from df2 to avoid duplication
            combined_row = tuple(left_row) + tuple(right_row[col] for col in df2.columns if col not in right_on)
            result.append(combined_row)

            # Mark the df2 row as having a join partner
            df2_join_partners[k] = True

        # Reset j to the previous position for the next iteration
        j = previous_j

    # Create the result DataFrame
    if result:
        # Define combined column names: df1 columns + df2 columns excluding right_on
        combined_columns = list(df1.columns) + [col for col in df2.columns if col not in right_on]
        result_np = np.array(result)
        result_frame = pd.DataFrame(result_np, columns=combined_columns)
    else:
        # If no results, return an empty DataFrame with combined columns
        combined_columns = list(df1.columns) + [col for col in df2.columns if col not in right_on]
        result_frame = pd.DataFrame(columns=combined_columns)

    # Handle different join types
    if join_type == 'inner':
        return result_frame
    elif join_type == 'outer':
        right_joined = handle_right_join(df1, df2, right_on, df1_join_partners)
        left_joined = handle_left_join(df1, df2, left_on, right_on, df2_join_partners)
        return pd.concat([result_frame, right_joined, left_joined], ignore_index=True)
    elif join_type == 'left':
        left_joined = handle_left_join(df1, df2, left_on, right_on, df2_join_partners)
        return pd.concat([result_frame, left_joined], ignore_index=True)
    elif join_type == 'right':
        right_joined = handle_right_join(df1, df2, right_on, df1_join_partners)
        return pd.concat([result_frame, right_joined], ignore_index=True)
    else:
        raise ValueError('Invalid join type')


def handle_right_join(df1, df2, right_on, df1_join_partners):
    """
    Handles the right join by selecting rows from df1 that have not been joined and adding NaNs for df2 columns.

    Parameters:
    - df1, df2: Original pandas DataFrames.
    - right_on: List of column names in df2 used for joining.
    - df1_join_partners: Boolean array indicating joined rows in df1.

    Returns:
    - A pandas DataFrame containing the right join results.
    """
    # Rows in df1 that have no join partners
    outer_rows = df1[~df1_join_partners].copy()
    # Add NaN for columns from df2
    new_columns = [col for col in df2.columns if col not in right_on]
    for col in new_columns:
        outer_rows[col] = np.nan
    return outer_rows


def handle_left_join(df1, df2, left_on, right_on, df2_join_partners):
    """
    Handles the left join by selecting rows from df2 that have not been joined and adding NaNs for df1 columns.

    Parameters:
    - df1, df2: Original pandas DataFrames.
    - left_on, right_on: Lists of column names used for joining.
    - df2_join_partners: Boolean array indicating joined rows in df2.

    Returns:
    - A pandas DataFrame containing the left join results.
    """
    # Rows in df2 that have no join partners
    outer_rows = df2[~df2_join_partners].copy()
    # Rename right_on columns to left_on names
    rename_mapping = {right: left for left, right in zip(left_on, right_on)}
    outer_rows.rename(columns=rename_mapping, inplace=True)
    # Add NaN for columns from df1
    new_columns = [col for col in df1.columns if col not in left_on]
    for col in new_columns:
        outer_rows[col] = np.nan
    return outer_rows


def test(size):
    """
    Test function to demonstrate the two_pointer_join functionality.

    Parameters:
    - size: Number of rows in the test DataFrames.
    """
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

    condition_left_below = lambda x, y: x['A'] <= y['C']
    condition_left_above = lambda x, y: x['B'] >= y['C']

    sorted_df = sortable_join(left_df, right_df, condition_left_below, condition_left_above, ['D'], ['E'], 'inner')
    print("Inner Join Result:")
    print(sorted_df)


if __name__ == "__main__":
    test(10)
