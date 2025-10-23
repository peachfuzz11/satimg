class ProductPathHandler:
    PATTERN = None

    def matches(self, pattern) -> bool:
        print(pattern, self.PATTERN.search(pattern))
        return self.PATTERN is not None and bool(self.PATTERN.search(pattern))
