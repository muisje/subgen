import queue

class UniqueLifoQueue(queue.LifoQueue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize)  # Initialize the base Queue class
        self._set = set()  # Set to track unique items ever added

    def put(self, item, block=True, timeout=None):
        # Only add the item if it hasn't been added before (for uniqueness)
        # Convert item to hashable type
        hashable_item = self._make_hashable(item)
        if hashable_item not in self._set:
            self._set.add(hashable_item)
            super().put(item, block, timeout)  # Call the parent put method

    def get(self, block=True, timeout=None):
        item = super().get(block, timeout)  # Call the parent get method
        return item

    def all_unique_items(self):
        return list(self._set)  # List of all unique items ever added
    
    def _make_hashable(self, item):
        """Convert mutable items (like dicts) to hashable types."""
        if isinstance(item, dict):
            # Convert dict to frozenset of its items to make it hashable
            return frozenset(item.items())
        return item  # For non-mutable items, return as is
