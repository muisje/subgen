from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
import inspect

@dataclass
class SubtitleEventConfig:
    """
    Configuration for a single callback function and its specific arguments.
    """
    callback: Callable
    specific_args: Dict[str, Any] = None
    state: Dict[str, Any] = field(default_factory=dict)  # Add a state dictionary

    def __post_init__(self):
        """Initialize specific_args to empty dict if None."""
        if self.specific_args is None:
            self.specific_args = {}

    def execute(self, shared_args: Dict[str, Any], **runtime_args) -> Any:
        """
        Execute the callback with all arguments combined.
        Priority: runtime_args > state > specific_args > shared_args
        """
        # Merge the state into the arguments
        args = {**shared_args, **self.specific_args, **self.state, **runtime_args}
        
        # Get the function signature
        signature = inspect.signature(self.callback)

        # Filter the args to only include parameters that are in the callback's signature
        valid_args = {
            param: value
            for param, value in args.items()
            if param in signature.parameters
        }
        
        # Execute the callback and store the result in the state
        result = self.callback(**valid_args)
        if result is not None:
            self.state.update(result)  # Update the state with the result
        
        return result


class SubtitleEventHandler:
    """
    Handles subtitle generation events with configurable callbacks and shared arguments.
    """
    def __init__(
        self,
        shared_args: Dict[str, Any] = None,
        on_start: Union[Callable, tuple[Callable, dict], List[Union[Callable, tuple[Callable, dict]]], None] = None,
        on_update: Union[Callable, tuple[Callable, dict], List[Union[Callable, tuple[Callable, dict]]], None] = None,
        on_detect_language: Union[Callable, tuple[Callable, dict], List[Union[Callable, tuple[Callable, dict]]], None] = None,
        on_complete: Union[Callable, tuple[Callable, dict], List[Union[Callable, tuple[Callable, dict]]], None] = None,
        on_error: Union[Callable, tuple[Callable, dict], List[Union[Callable, tuple[Callable, dict]]], None] = None,
        on_skip: Union[Callable, tuple[Callable, dict], List[Union[Callable, tuple[Callable, dict]]], None] = None,
        on_progress: Union[Callable, tuple[Callable, dict], List[Union[Callable, tuple[Callable, dict]]], None] = None  # Add on_progress
    ):
        """
        Initialize the handler with callbacks and arguments.
        
        Args:
            shared_args: Arguments shared between all callbacks (e.g., api_key, base_url)
            on_start: Single callback, tuple, or list of callbacks for start event
            on_update: Single callback, tuple, or list of callbacks for update event
            on_detect_language: Single callback, tuple, or list of callbacks for language detection event
            on_complete: Single callback, tuple, or list of callbacks for completion event
            on_error: Single callback, tuple, or list of callbacks for error event
            on_skip: Single callback, tuple, or list of callbacks for skip event
            on_progress: Single callback, tuple, or list of callbacks for progress event
        """
        self._shared_args = shared_args or {}
        
        # Convert all callbacks to SubtitleEventConfig objects
        self._on_start = self._create_configs(on_start)
        self._on_update = self._create_configs(on_update)
        self._on_detect_language = self._create_configs(on_detect_language)
        self._on_complete = self._create_configs(on_complete)
        self._on_error = self._create_configs(on_error)
        self._on_skip = self._create_configs(on_skip)
        self._on_progress = self._create_configs(on_progress)  # Add on_progress

    def _create_config(self, item) -> Optional[SubtitleEventConfig]:
        """
        Convert a callback item to SubtitleEventConfig.
        
        Args:
            item: Can be:
                - None: returns None
                - Callable: creates SubtitleEventConfig with no specific args
                - Tuple[Callable, dict]: creates SubtitleEventConfig with specific args
        """
        if item is None:
            return None
        if callable(item):
            return SubtitleEventConfig(item)
        if isinstance(item, tuple):
            return SubtitleEventConfig(item[0], item[1])
        return None

    def _create_configs(self, items) -> List[SubtitleEventConfig]:
        """
        Convert a list of callback items to a list of SubtitleEventConfigs.
        
        Args:
            items: Can be:
                - None: returns empty list
                - Callable: wraps in a list and creates SubtitleEventConfig
                - Tuple[Callable, dict]: wraps in a list and creates SubtitleEventConfig
                - List[Union[Callable, tuple[Callable, dict]]]: processes each item
        """
        if not items:
            return []
        
        # If a single function or tuple is passed, wrap it in a list
        if callable(items) or isinstance(items, tuple):
            items = [items]
        
        return [self._create_config(item) for item in items]

    def _execute_event(self, configs: Union[SubtitleEventConfig, List[SubtitleEventConfig]], **kwargs) -> List[Any]:
        """
        Execute one or more event callbacks with provided arguments.
        """
        if not configs:
            return []
        if isinstance(configs, SubtitleEventConfig):
            return [configs.execute(shared_args=self._shared_args, **kwargs)]
        return [config.execute(shared_args=self._shared_args, **kwargs) 
                for config in configs]

    def on_start(self, **kwargs) -> List[Any]:
        """Execute start callback(s) with runtime arguments."""
        return self._execute_event(self._on_start, **kwargs)
    
    def on_update(self, **kwargs) -> List[Any]:
        """Execute all update callbacks with runtime arguments."""
        return self._execute_event(self._on_update, **kwargs)

    def on_detect_language(self, **kwargs) -> List[Any]:
        """Execute all language detection callbacks with runtime arguments."""
        return self._execute_event(self._on_detect_language, **kwargs)
    
    def on_complete(self, **kwargs) -> List[Any]:
        """Execute finish callback(s) with runtime arguments."""
        return self._execute_event(self._on_complete, **kwargs)
    
    def on_skip(self, **kwargs) -> List[Any]:
        """Execute skip callback(s) with runtime arguments."""
        return self._execute_event(self._on_skip, **kwargs)
    
    def on_error(self, **kwargs) -> List[Any]:
        """Execute error callback(s) with runtime arguments."""
        return self._execute_event(self._on_error, **kwargs)
    
    def on_progress(self, **kwargs) -> List[Any]:
        """Execute progress callback(s) with runtime arguments."""
        return self._execute_event(self._on_progress, **kwargs)