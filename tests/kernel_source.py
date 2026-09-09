"""Small source extractor for production-kernel host oracles."""

def extract_function(source, signature):
    if source.count(signature) != 1:
        raise AssertionError(f"expected one production definition: {signature}")
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise AssertionError(f"unterminated production definition: {signature}")
    return source[start:end]
