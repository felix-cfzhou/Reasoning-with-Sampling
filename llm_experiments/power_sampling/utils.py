"""Small list helper used by the SMC and TMC samplers."""


def unflatten_list(flattened_list, lengths):
    """
    Takes a flattened list and a list of lengths to reconstruct
    the original list of lists.
    """
    unflattened = []
    start = 0
    for length in lengths:
        unflattened.append(flattened_list[start : start + length])
        start += length
    return unflattened
