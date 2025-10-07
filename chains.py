from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Chain:
    """单条链的配置类"""
    name: str
    id: int
    aliases: List[str]  # 别名列表，如 bsc/bnb, eth/ethereum 等
    
    def matches(self, name_or_alias: str) -> bool:
        """检查名称或别名是否匹配"""
        name_or_alias = name_or_alias.lower()
        return (name_or_alias == self.name.lower() or 
                name_or_alias in [a.lower() for a in self.aliases])

class Chains:
    """链集合管理类"""
    def __init__(self, chains: List[Chain]):
        self.chains = chains
        
    def get_chain(self, name_or_alias: str) -> Optional[Chain]:
        """通过名称或别名查找链"""
        for chain in self.chains:
            if chain.matches(name_or_alias):
                return chain
        return None
    
    def __iter__(self):
        return iter(self.chains)
    
    def __len__(self):
        return len(self.chains)

# 预定义链别名映射
CHAIN_ALIASES = {
    "ethereum": ["eth"],
    "binance": ["bsc", "bnb"],
    "avalanche": ["avax"],
    "polygon": ["matic"],
    "optimism": ["op"],
    "arbitrum": ["arb"],
}

def load_chains_from_config(config: dict) -> Chains:
    """从配置文件加载链信息"""
    chain_list = []
    
    for chain_data in config.get("chains", []):
        name = chain_data["name"]
        # 获取预定义别名
        aliases = CHAIN_ALIASES.get(name, [])
        # 添加 kyber_slug 作为别名(如果存在且不同于name)
        if "kyber_slug" in chain_data and chain_data["kyber_slug"] != name:
            aliases.append(chain_data["kyber_slug"])
            
        chain = Chain(
            name=name,
            id=chain_data["chain_id"],
            aliases=aliases
        )
        chain_list.append(chain)
        
    return Chains(chain_list)