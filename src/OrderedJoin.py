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
