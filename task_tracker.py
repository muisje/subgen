import threading
import uuid

class TaskTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._internal_lock = threading.Lock()
                cls._instance.pending_tasks = {}  # {task_group_id: task_count}
            return cls._instance

    def track_new_tasks(self):
        """Generate a unique UUID to track a group of tasks and return it."""
        with self._internal_lock:
            task_group_id = uuid.uuid4()  # Universally unique ID
            self.pending_tasks[task_group_id] = 0
            return task_group_id

    def add_task(self, task_group_id):
        with self._internal_lock:
            if task_group_id not in self.pending_tasks:
                raise ValueError(f"Task group ID {task_group_id} is not being tracked.")
            self.pending_tasks[task_group_id] += 1

    def task_finished(self, task_group_id):
        with self._internal_lock:
            if task_group_id not in self.pending_tasks:
                return
            self.pending_tasks[task_group_id] -= 1
            if self.pending_tasks[task_group_id] <= 0:
                del self.pending_tasks[task_group_id]
                
    def tasks_remaining(self, task_group_id):
        with self._internal_lock:
            return self.pending_tasks.get(task_group_id, 0)

    def did_all_tasks_finish(self, task_group_id):
        with self._internal_lock:
            return task_group_id not in self.pending_tasks