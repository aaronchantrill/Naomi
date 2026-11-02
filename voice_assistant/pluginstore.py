import importlib
import inspect
import json
import os


MANDATORY_OPTIONS: set[set] = (
    ('Plugin', 'Name'),
    ('Plugin', 'Version'),
    ('Plugin', 'URL')
)


def check_keys(struct: dict, keys: set) -> bool:
    """
    Checks if the sequence of keys defined in 'keys' exists in the nested
    dictionary 'struct'.

    Args:
        struct: the dictionary to search
        keys: the tuple of keys representing the path to search

    Returns
        True if the path exists, False otherwise
    """
    current = struct
    for key in path:
        if not isinstance(current, dict):
            return False
        if key not in current:
            return False
        current = current[key]
    return True


def parse_info_file(infofile_path: str) -> dict:
    info: dict = dict()
    with open(infofile_path) as f:
        info = json.loads(f.read())
        options_missing: bool = False
        for option in MANDATORY_OPTIONS:
            if not check_keys(info, *option):
                options_missing = True
                logger.warn(
                    f"Plugin info file '{infofile_path}' missing value '{option}'"
                )
        # Check to see if the LICENSE[.txt|.md] file exists in the plugin directory
        licensefile_path = os.path.join(os.path.dirname(infofile_path), 'LICENSE')
        if( not os.path.isfile(licensefile_path)
            and not os.path.isfile(f"{licensefile_path}.txt")
            and not os.path.isfile(f"{licensefile_path}.md")
            and not check_keys(info, ('Plugin', 'License'))
        ):
            options_missing = True
            logger.debug(f"Missing license. Add file {licensefile_path}.")
        if options_missing:
            raise PluginError("Info file is missing values!")
    logger.debug(f"Plugin info file {infofile_path} parsed successfully")
    return info


class PluginInfo():
    def __init__(self, info, plugin_class, translations, directory) -> None:
        self._info = info
