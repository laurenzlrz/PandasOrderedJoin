from OrderedJoin import OrderedJoin


class KeysUnsortedOrderedJoin(OrderedJoin):

    def __init__(self, left_df, right_df, condition_left_below, condition_right_below, right_on, left_on):
        super().__init__(left_df, right_df, condition_left_below, condition_right_below, right_on, left_on)

    def iterate_until_not(self):
        for current_right_row in range(self.next_row_start, len(self.right_np)):

            if not self.condition_left_below(self.left_np[self.current_left_row], self.right_np[current_right_row]):
                self.next_row_start += 1
                continue
            if not self.condition_right_below(self.left_np[self.current_left_row], self.right_np[current_right_row]):
                break

            self.new_concat_rows_numpy(self.left_np[self.current_left_row], self.right_np[current_right_row])

    def get_outer(self):

        right_result = self.check_outer_right_columns(self.right_np)
        left_result = self.check_outer_left_columns(self.left_np)

        self.outer_right = right_result[~right_result[self.left_on].isin(self.left_np[self.left_on])]
        self.outer_left = left_result[~left_result[self.right_on].isin(self.right_np[self.left_on])]
