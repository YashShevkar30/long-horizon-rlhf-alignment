def calculate_accuracy(predictions, labels):
    return sum([p == l for p, l in zip(predictions, labels)]) / len(labels)
