

from typing import Any, Dict,List
from attr import dataclass
from chains import Chain
from assets import Assets


@dataclass
class AppConfig:
    api: Dict[str, Any]
    chains: List[Chain]
    assets: Assets  # Change this from Dict[str, Asset] to Assets
    settings: Dict[str, Any]