from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
import inspect

@dataclass
class SubtitleEventConfig:
    callback: Callable
    specific_args: Dict[str, Any] = None
    state: Dict[str, Any] = field(default_factory=dict)
    merge_order: List[str] = field(
        default_factory=lambda: [
            "runtime_args",    # Highest priority (takes precedence over all others)
            "specific_args",   # Second priority
            "state",           # Third priority
            "shared_args",     # Fourth priority
            "shared_state"     # Lowest priority (first to be overwritten)
        ]
    )
    save_own_state: bool = False
    save_shared_state: bool = False

    def __post_init__(self):
        if self.specific_args is None:
            self.specific_args = {}

    def execute(self, shared_args: Dict[str, Any], shared_state: Dict[str, Any], **runtime_args) -> Any:
        # Initialize argument sources with correct priority levels
        argument_sources = {
            "runtime_args": runtime_args,
            "specific_args": self.specific_args,
            "state": self.state,
            "shared_args": shared_args,
            "shared_state": shared_state,
        }

        # Merge arguments according to specified priority
        merged_args = {}
        for source_name in self.merge_order:
            if source_name in argument_sources:
                merged_args.update(argument_sources[source_name])

        # Filter to only parameters the callback actually accepts
        sig_params = inspect.signature(self.callback).parameters
        filtered_args = {
            param: merged_args[param]
            for param in sig_params
            if param in merged_args
        }

        result = self.callback(**filtered_args)

        # Handle state updates based on configuration
        if isinstance(result, dict):
            if self.save_own_state:
                self.state.update(result)
            if self.save_shared_state:
                shared_state.update(result)

        return result

class SubtitleEventHandler:
    def __init__(
        self,
        shared_args: Dict[str, Any] = None,
        on_start: Union[Callable, tuple[Callable, dict, list], List[Union[Callable, tuple[Callable, dict, list]]], None] = None,
        on_update: Union[Callable, tuple[Callable, dict, list], List[Union[Callable, tuple[Callable, dict, list]]], None] = None,
        on_detect_language: Union[Callable, tuple[Callable, dict, list], List[Union[Callable, tuple[Callable, dict, list]]], None] = None,
        on_detect_language_failed: Union[Callable, tuple[Callable, dict, list], List[Union[Callable, tuple[Callable, dict, list]]], None] = None,
        on_complete: Union[Callable, tuple[Callable, dict, list], List[Union[Callable, tuple[Callable, dict, list]]], None] = None,
        on_error: Union[Callable, tuple[Callable, dict, list], List[Union[Callable, tuple[Callable, dict, list]]], None] = None,
        on_skip: Union[Callable, tuple[Callable, dict, list], List[Union[Callable, tuple[Callable, dict, list]]], None] = None,
        on_progress: Union[Callable, tuple[Callable, dict, list], List[Union[Callable, tuple[Callable, dict, list]]], None] = None  # Add on_progress
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
        self.shared_state = {}  # Owner of shared state

        # Convert all callbacks to SubtitleEventConfig objects
        self._on_start = self._create_configs(on_start)
        self._on_update = self._create_configs(on_update)
        self._on_detect_language = self._create_configs(on_detect_language)
        self._on_detect_language_failed = self._create_configs(on_detect_language_failed)
        self._on_complete = self._create_configs(on_complete)
        self._on_error = self._create_configs(on_error)
        self._on_skip = self._create_configs(on_skip)
        self._on_progress = self._create_configs(on_progress)  # Add on_progress

    def _create_config(self, item) -> Optional[SubtitleEventConfig]:
        if item is None:
            return None
        if isinstance(item, SubtitleEventConfig):
            return item
        if callable(item):
            return SubtitleEventConfig(item)
        if isinstance(item, tuple):
            return SubtitleEventConfig(
                callback=item[0],
                specific_args=item[1] if len(item) > 1 else None,
                merge_order=item[2] if len(item) > 2 else None,
                save_own_state=item[3] if len(item) > 3 else False,
                save_shared_state=item[4] if len(item) > 4 else False,
            )
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
        if callable(items) or isinstance(items, tuple) or isinstance(items, SubtitleEventConfig):
            items = [items]
        
        return [self._create_config(item) for item in items]

    def _execute_event(self, configs: Union[SubtitleEventConfig, List[SubtitleEventConfig]], **kwargs) -> List[Any]:
        """
        Execute one or more event callbacks with provided arguments.
        """
        if not configs:
            return []
        if isinstance(configs, SubtitleEventConfig):
            return [configs.execute(shared_args=self._shared_args, shared_state=self.shared_state, **kwargs)]
        return [config.execute(shared_args=self._shared_args, shared_state=self.shared_state, **kwargs) 
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

    def on_detect_language_failed(self, **kwargs) -> List[Any]:
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