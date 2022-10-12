def check_sorted(l1, l2):
    not_sorted = False
    for val, lab in zip(l1, l2):
        if not (val[0] in lab or lab[0] in val):
            not_sorted = True
        if val[-1] in lab or lab[-1] in val:
            not_sorted = False
    return not_sorted


def fix_order(l1, l2):
    new_list1 = [[] for _ in l1]
    for value in l1:
        pos = [i for i, v2 in enumerate(l2) for v1 in value if v1 in v2]
        for p in pos:
            new_list1[p] = value
    if [] not in new_list1:
        return new_list1
    else:
        print('ER')


def get_lab_type(df):
    if df['label'].nunique() == 1:
        if df['label'].all():
            return 1
        else:
            return 0
    else:
        return 2
