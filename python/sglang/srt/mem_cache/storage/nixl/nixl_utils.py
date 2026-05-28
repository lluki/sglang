import logging
import os
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


_STORAGE_LIMIT_UNITS = {"mb": 1024**2, "gb": 1024**3, "tb": 1024**4}


def parse_storage_limit(value) -> int:
    """Strictly-positive bytes int, or '<N>mb|gb|tb' (case-insensitive)."""
    if isinstance(value, int) and not isinstance(value, bool):
        result = value
    elif isinstance(value, str):
        s = value.strip().lower()
        result = None
        for unit, mult in _STORAGE_LIMIT_UNITS.items():
            if s.endswith(unit):
                num = s[: -len(unit)].strip()
                if not num:
                    raise ValueError(
                        f"invalid storage_limit: missing number before {unit!r}"
                    )
                result = int(num) * mult
                break
        if result is None:
            result = int(s)
    else:
        raise ValueError(f"invalid storage_limit: {value!r}")
    if result <= 0:
        raise ValueError(
            f"invalid storage_limit: must be > 0, got {result} (from {value!r})"
        )
    return result


class NixlBackendConfig:
    """Handles NIXL backend configurations"""

    def __init__(self, config: Optional[dict[str, str]] = None):
        """Initialize backend configuration.
        Args:
            config: configurations in a dictionary. This config comes from --hicache-storage-backend-extra-config

            config can be in two forms:
            1. fully qualified form (for all plugins, some of them are enabled, others not):
                {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            2. flat form (for a specific selected plugin), assuming all params apply to a selected plugin
                {'param1': 'value1', 'param2': 'value2', ...}
        """
        self.config = config or {}

    def get_storage_limit_bytes(self) -> Optional[int]:
        raw = self.config.get("storage_limit")
        if raw is None and isinstance(self.config.get("plugin"), dict):
            for plugin_cfg in self.config["plugin"].values():
                if isinstance(plugin_cfg, dict) and "storage_limit" in plugin_cfg:
                    raw = plugin_cfg["storage_limit"]
                    break
        if raw is None:
            return None
        return parse_storage_limit(raw)

    def get_use_direct_io(self) -> bool:
        """Return True if O_DIRECT should be requested when opening files.

        Checks the top-level ``use_direct_io`` key in the long-form JSON config first,
        then falls back to the ``SGLANG_HICACHE_NIXL_USE_DIRECT_IO`` environment variable
        (default: enabled).
        """
        if "use_direct_io" in self.config:
            return bool(self.config["use_direct_io"])
        return envs.SGLANG_HICACHE_NIXL_USE_DIRECT_IO.get()

    def get_specified_plugin(self) -> str:
        """decide which plugin to use: either config or SGLANG_HICACHE_NIXL_BACKEND_PLUGIN specifies the plugin, if not, use "auto" """

        if "plugin" in self.config:
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            # choose the FIRST active plugin
            for key, item in self.config["plugin"].items():
                if item.get("active", False) in [True, "true", "True"]:
                    plugin = key.upper()
                    break
        else:
            # config is empty, or in flat form {'param1': 'value1', 'param2': 'value2', ...}
            plugin = os.getenv("SGLANG_HICACHE_NIXL_BACKEND_PLUGIN", "auto")

        return plugin

    def get_backend_initparams(self, backend_name) -> dict:
        """Get initialization parameters from config of NIXL backend for backend creation.
        Args:
            backend_name: a specific backend's name (already converted "auto" into a specific backend name)

        """

        initparams = {}

        # config can be in two forms:
        if "plugin" in self.config:
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            if backend_name.lower() in self.config["plugin"]:
                config_data = self.config["plugin"][backend_name.lower()]
            else:
                logger.debug(
                    f"No specific config found for plugin {backend_name} in extra_config. Use default init params."
                )
                config_data = {}
        else:
            # flat form {'param1': 'value1', 'param2': 'value2', ...}
            config_data = self.config

        for key, value in config_data.items():
            initparams[key] = str(value)

        return initparams


class NixlBackendSelection:
    """Handles NIXL backend selection and creation."""

    # Priority order for File-based plugins in case of auto selection
    FILE_PLUGINS = ["3FS", "POSIX", "GDS_MT", "GDS"]
    # Priority order for File-based plugins in case of auto selection (add more as needed)
    OBJ_PLUGINS = ["OBJ"]  # Based on Amazon S3 SDK

    def __init__(
        self, plugin: str = "auto", nixlconfig: Optional[NixlBackendConfig] = None
    ):
        """Initialize backend selection.
        Args:
            plugin: Plugin to use (default "auto" selects best available).
                   Can be a file plugin (3FS, POSIX, GDS, GDS_MT) or
                   an object plugin (OBJ).
        """
        self.plugin = plugin
        self.backend_name = None
        self.mem_type = None
        self.nixlconfig = nixlconfig

    def create_backend(self, agent) -> bool:
        """Create the appropriate NIXL backend based on configuration."""
        try:
            plugin_list = agent.get_plugin_list()
            logger.debug(f"Available NIXL plugins: {plugin_list}")

            # Handle explicit plugin selection or auto priority
            if self.plugin == "auto":
                # Try all file plugins first
                for plugin in self.FILE_PLUGINS:
                    if plugin in plugin_list:
                        self.backend_name = plugin
                        break
                # If no file plugin found, try object plugins
                if not self.backend_name:
                    for plugin in self.OBJ_PLUGINS:
                        if plugin in plugin_list:
                            self.backend_name = plugin
                            break
            else:
                # Use explicitly requested plugin
                self.backend_name = self.plugin

            if self.backend_name not in plugin_list:
                logger.error(
                    f"Backend {self.backend_name} not available in plugins: {plugin_list}"
                )
                return False

            # obtain initparams for the backend from the NIXL config
            initparams = (
                self.nixlconfig.get_backend_initparams(self.backend_name)
                if self.nixlconfig
                else {}
            )

            # Create backend and set memory type
            if self.backend_name in self.OBJ_PLUGINS and "bucket" not in initparams:
                bucket = os.environ.get("AWS_DEFAULT_BUCKET")
                if not bucket:
                    logger.error(
                        "AWS_DEFAULT_BUCKET environment variable must be set for object storage"
                    )
                    return False

                initparams["bucket"] = bucket

            # create backend using initialization parameters
            agent.create_backend(self.backend_name, initparams)

            logger.info(
                f"NixlBackendSelection.create_backend: backend_name {self.backend_name} initparams {initparams} customParams {agent.get_backend_params(self.backend_name)} supported plugins {plugin_list}"
            )

            self.mem_type = "OBJ" if self.backend_name in self.OBJ_PLUGINS else "FILE"
            logger.debug(
                f"Created NIXL backend: {self.backend_name} with memory type: {self.mem_type}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to create NIXL backend: {e}, backend_name {self.backend_name}, supported plugins {plugin_list} initparams {initparams}"
            )
            return False


class NixlFileManager:
    """Handles file system operations for NIXL."""

    def __init__(self, base_dir: str, use_direct_io: bool = True):
        """
        Initialize file manager.
        Args:
            base_dir: Base directory for storing tensor files
            use_direct_io: If True, open files with O_DIRECT (bypasses OS page cache).
                Falls back to buffered I/O with a warning when O_DIRECT is unavailable.
        """
        self.base_dir = base_dir
        self.use_direct_io = use_direct_io
        if base_dir == "":
            logger.debug(
                f"Initialized file manager without a base directory. Direct I/O: {use_direct_io}"
            )
        else:
            os.makedirs(base_dir, exist_ok=True)
            logger.debug(
                f"Initialized file manager with base directory: {base_dir}. Direct I/O: {use_direct_io}"
            )

    def clear(self) -> None:
        """Clear all files in the base directory."""
        if self.base_dir == "":
            logger.warning("Base directory is empty, skipping clear operation")
            return

        try:
            for root, dirs, files in os.walk(self.base_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
            logger.debug(f"Cleared all files in base directory: {self.base_dir}")
        except Exception as e:
            logger.error(
                f"Failed to clear files in base directory {self.base_dir}: {e}"
            )

    def get_file_path(self, key: str) -> str:
        """Get full file path for a given key."""
        return os.path.join(self.base_dir, key)

    def open_file(self, file_path: str, create: bool = False) -> Optional[int]:
        """Open a file and return its file descriptor.

        If ``create`` is True, the file is created if it does not exist
        (mode 0o644, no truncation). When ``self.use_direct_io`` is True,
        the file is opened with ``O_DIRECT`` (bypasses the OS page cache);
        falls back to buffered I/O with a warning if ``O_DIRECT`` is
        unavailable on this platform.
        """
        flags = os.O_RDWR | os.O_CREAT if create else os.O_RDWR
        if self.use_direct_io:
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            else:
                logger.warning(
                    "use_direct_io is True, but O_DIRECT is not available on "
                    "this system. Falling back to buffered I/O."
                )
        try:
            return os.open(file_path, flags, 0o644)
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            return None

    def close_file(self, fd: int) -> bool:
        """Close a file descriptor."""
        try:
            os.close(fd)
            return True
        except Exception as e:
            logger.error(f"Failed to close file descriptor {fd}: {e}")
            return False
