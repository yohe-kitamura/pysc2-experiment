def transformDistance(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
        return [x - x_distance, y - y_distance]

    return [x + x_distance, y + y_distance]


def transformLocation(self, x, y):
    if not self.base_top_left:
        return [64 - x, 64 - y]

    return [x, y]

