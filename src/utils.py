import re


def re_findall(pattern, string):
    return [m.groupdict() for m in re.finditer(pattern, string)]


def jaccard(x1, x2, y1, y2):
    # Calculate jaccard index
    intersection = max(0, min(x2, y2)-max(x1, y1))
    filled_union = max(x2, y2) - min(x1, y1)
    return intersection/filled_union if filled_union > 0 else 0


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    return match.group(group) if match else default
