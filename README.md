# Satellite Products

Sat-products is a wrapper for satellite data and provides a simple interface for reading different products.

The project automatically attempts to create a wrapper based on a provided path.
Instantiation is split in two phases:

1. Product resolution
2. Product initialization

The first phase attempts to resolve the input to decide which product to instantiate.
This phase throws a ProductResolutionError if unsuccessful.

The second phase validates the product e.g. that it contains the expected data/metadata.
This phase throws a ProductInitializationError if unsuccessful.