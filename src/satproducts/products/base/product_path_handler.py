class ProductPathHandler:
    PATTERN = None

    def matches(self, pattern) -> bool:
        return self.PATTERN is not None and bool(self.PATTERN.search(pattern))
