import numpy as np

from AbstractOrderedJoin import OrderedJoin


class KeysSortedOrderedJoin(OrderedJoin):

    def __init__(self, left_df, right_df, condition_left_below, condition_right_below, right_on, left_on):
        super().__init__(left_df, right_df, condition_left_below, condition_right_below, right_on, left_on)
        self.current_right_row = 0

        self.right_inner_rows = np.full(len(right_df), False, dtype=bool)
        self.left_inner_rows = np.full(len(left_df), False, dtype=bool)

    def iterate_until_not(self):
        right_row_number = self.next_row_start

        while right_row_number < len(self.right_np):
            right_row_number += 1
            if not self.join_below_condition(self.left_np[self.current_left_row], self.right_np[right_row_number]):
                self.next_row_start = right_row_number
                continue
            if not self.join_above_condition(self.left_np[self.current_left_row], self.right_np[right_row_number]):
                break

            self.right_inner_rows[right_row_number] = True
            self.left_inner_rows[self.current_left_row] = True

            if not self.condition_left_below(self.left_np[self.current_left_row], self.right_np[right_row_number]):
                self.next_row_start = right_row_number
                continue
            if not self.condition_right_below(self.left_np[self.current_left_row], self.right_np[right_row_number]):
                break

            self.new_concat_rows_numpy(self.left_np[self.current_left_row], self.right_np[right_row_number])

    def join_below_condition(self, x, y):
        x_keys = np.array(x[self.left_on].tolist())
        y_keys = np.array(y[self.right_on].tolist())
        return np.all(x_keys <= y_keys)

    def join_above_condition(self, x, y):
        x_keys = np.array(x[self.left_on].tolist())
        y_keys = np.array(y[self.right_on].tolist())
        return np.all(x_keys >= y_keys)

    def determine_outer(self):
        self.outer_left_to_check = self.left_np[~self.left_inner_rows]
        self.outer_right_to_check = self.right_np[~self.right_inner_rows]

        self.outer_right = self.check_outer_right_columns()
        self.outer_left = self.check_outer_left_columns()
