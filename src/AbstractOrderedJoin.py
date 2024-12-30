from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# SolvedTodo: make join with condition clear
# SolvedTodo: Then make the filters clear
# SolvedTODO: check if assumption with sorted data is correct
# Todo: Improve the code quality of the following code
# Todo: Clear numpy usage
# Todo: Cleaning of the classes and better import structure.
# Todo: Testing

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


class OrderedJoin(ABC):
    def __init__(self, left_df, right_df, condition_left_below, condition_right_below, right_on, left_on):
        self.result = []
        self.current_left_row = 0
        self.next_row_start = 0

        self.left_np = left_df.to_records(index=False)
        self.right_np = right_df.to_records(index=False)
        self.condition_left_below = condition_left_below
        self.condition_right_below = condition_right_below
        self.right_on = right_on
        self.left_on = left_on

        self.outer_left = None
        self.outer_right = None
        self.outer_right_to_check = None
        self.outer_left_to_check = None

    @abstractmethod
    def iterate_until_not(self):
        pass

    @abstractmethod
    def determine_outer(self):
        pass

    def right_join(self):
        self.inner_join()
        self.determine_outer()
        return pd.concat([pd.DataFrame(self.result), self.outer_right], ignore_index=True)

    def left_join(self):
        self.inner_join()
        self.determine_outer()
        return pd.concat([pd.DataFrame(self.result), self.outer_left], ignore_index=True)

    def outer_join(self):
        self.inner_join()
        self.determine_outer()
        return pd.concat([pd.DataFrame(self.result), self.outer_left, self.outer_right], ignore_index=True)

    def inner_join(self):
        for i in range(0, len(self.left_np)):
            self.current_left_row = i
            self.iterate_until_not()

    def join_equal_condition(self, x, y):
        x_keys = np.array(x[self.left_on].tolist())
        y_keys = np.array(y[self.right_on].tolist())
        return np.all(x_keys <= y_keys)

    # TODO: Right now return as a dictionary
    def new_concat_rows_numpy(self, left_np_row, right_np_row):

        if not self.join_equal_condition(left_np_row, right_np_row):
            return None

        combined_row = {}
        for col in left_np_row.dtype.names:
            combined_row[col] = left_np_row[col]
        for col in right_np_row.dtype.names:
            if col not in self.right_on:
                combined_row[col] = right_np_row[col]
        self.result.append(combined_row)

    def check_outer_right_columns(self):
        result = []
        left_nan_row = pd.Series(data=np.nan, index=self.left_np.columns)

        for right_row in range(0, len(self.outer_right_to_check)):
            if (self.condition_left_below(left_nan_row, self.outer_right_to_check[right_row]) and
                    self.condition_right_below(left_nan_row, self.outer_right_to_check[right_row])):
                result.append(pd.concat([left_nan_row, self.outer_right_to_check[right_row]]))

        result_df = pd.DataFrame(result).drop(columns=self.left_on)
        rename_dict = dict(zip(self.right_on, self.left_on))
        return result_df.rename(columns=rename_dict)

    def check_outer_left_columns(self):
        result = []
        right_nan_row = pd.Series(data=np.nan, index=self.right_np.columns)
        for left_row in range(0, len(self.outer_left_to_check)):
            if (self.condition_left_below(self.outer_left_to_check[left_row], right_nan_row)
                    and self.condition_right_below(self.outer_left_to_check[left_row], right_nan_row)):
                result.append(self.outer_left_to_check[left_row])
        return pd.DataFrame(result).drop(columns=self.right_on)
