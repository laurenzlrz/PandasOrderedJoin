from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from KeysSortedOrderedJoin import KeysSortedOrderedJoin
from KeysUnsortedOrderedJoin import KeysUnsortedOrderedJoin


def ordered_join(left_df, right_df, condition_left_below, condition_right_below, right_on, left_on, join_type='inner'):

    try:
        left_df = left_df.sort_values(by=left_on, kind='stable').reset_index(drop=True)
        right_df = right_df.sort_values(by=right_on, kind='stable').reset_index(drop=True)
        join_executor = KeysSortedOrderedJoin(left_df, right_df, condition_left_below,
                                              condition_right_below, right_on, left_on)
    except TypeError:
        join_executor = KeysUnsortedOrderedJoin(left_df, right_df, condition_left_below, condition_right_below,
                                                right_on, left_on)

    if join_type == 'inner':
        return join_executor.inner_join()
    if join_type == 'right':
        return join_executor.right_join()
    if join_type == 'left':
        return join_executor.left_join()
    if join_type == 'outer':
        return join_executor.outer_join()
    return Exception('Invalid join type')


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

    @abstractmethod
    def iterate_until_not(self):
        pass

    @abstractmethod
    def get_outer(self):
        pass

    def right_join(self):
        self.inner_join()
        self.get_outer()
        return pd.concat([pd.DataFrame(self.result), self.outer_right], ignore_index=True)

    def left_join(self):
        self.inner_join()
        self.get_outer()
        return pd.concat([pd.DataFrame(self.result), self.outer_left], ignore_index=True)

    def outer_join(self):
        self.inner_join()
        self.get_outer()
        return pd.concat([pd.DataFrame(self.result), self.outer_left, self.outer_right], ignore_index=True)

    def inner_join(self):
        for i in range(0, len(self.left_np)):
            self.current_left_row = i
            self.iterate_until_not()

    def join_equal_condition(self, x, y):
        x_keys = np.array(x[self.left_on].tolist())
        y_keys = np.array(y[self.right_on].tolist())
        return np.all(x_keys <= y_keys)

    # Right now return as a dictionary
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

    def check_outer_right_columns(self, right_df):
        result = []
        left_nan_row = pd.Series(data=np.nan, index=self.left_np.columns)
        for right_row in range(0, len(right_df)):
            if (self.condition_left_below(left_nan_row, right_df[right_row]) and
                    self.condition_right_below(left_nan_row, right_df[right_row])):
                result.append(pd.concat([left_nan_row, right_df[right_row]]))
        result_df = pd.DataFrame(result).drop(columns=self.left_on)
        rename_dict = dict(zip(self.right_on, self.left_on))
        return result_df.rename(columns=rename_dict)

    def check_outer_left_columns(self, left_df):
        result = []
        right_nan_row = pd.Series(data=np.nan, index=self.right_np.columns)
        for left_row in range(0, len(left_df)):
            if self.condition_left_below(left_df[left_row], right_nan_row) and self.condition_right_below(
                    left_df[left_row], right_nan_row):
                result.append(left_df[left_row])
        return pd.DataFrame(result).drop(columns=self.right_on)
