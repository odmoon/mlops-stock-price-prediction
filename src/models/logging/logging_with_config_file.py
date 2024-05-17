import logging
import logging.config
from rich.logging import RichHandler
from pathlib import Path

logging.config.fileConfig(Path(__file__).resolve().parent.parent.parent / "log_config.config")
logger = logging.getLogger(__name__)
logger.root.handlers[0] = RichHandler(markup = True)