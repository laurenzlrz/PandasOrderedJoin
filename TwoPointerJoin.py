import numpy as np
import pandas as pd

def join(left_df, right_df):
    j = 0
    result = []
    df1_join_partners = np.full(len(left_df), False, dtype=bool)
    df2_join_partners = np.full(len(right_df), False, dtype=bool)

    for i in range(len(left_df)):
        left_row = left_df[i]

        # Means did not find any row to join on. Belongs to outer join, if not previously joined.
        # No need to differentiata because join partners set in later loop.
        while j < len(right_df) and not joinon_left_below(left_row, right_df[j]) or not condition_left_below(left_row, right_df[j]):
            j += 1

        if j >= len(right_df):
            break  # No more matches possible

        # Now its from j joinable
        # Means did not find any row to join on. Belongs to outer join.

        # Store the starting position to reset j later
        previous_j = j

        # Iterate over df2 starting from j
        for k in range(j, len(right_df)):
            right_row = right_df[k]

            # Check if the row satisfies the upper condition
            if not joinon_right_below(left_row, right_row):
                break  # Exit the inner loop as further rows won't satisfy the condition

            df1_join_partners[i] = True
            df2_join_partners[k] = True

            # Means we failed to join because of the condition
            if not condition_right_below(left_row, right_row):
                break

            # Concatenate the rows based on the join keys
            concatenated = new_concat_rows_numpy(
                left_df, right_df, i, k, left_on, right_on
            )
            if concatenated is not None:
                result.append(concatenated)

            # Mark the df2 row as having a join partner

        # Reset j to the previous position for the next iteration
        j = previous_j