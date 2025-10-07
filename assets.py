from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from chains import Chain, Chains
from logger_utils import NullLogger

@dataclass
class Token:
    """代币在特定链上的实例"""
    name: str          # 代币名称/符号，如 "USDT"
    chain: Chain       # 所在链的 Chain 对象
    address: str       # 合约地址
    decimals: int      # 精度(小数位数)
    
    def __str__(self) -> str:
        return f"{self.name}@{self.chain.name}"
    
    def __repr__(self) -> str:
        return f"Token({self.name}, chain={self.chain.name}, addr={self.address}, decimals={self.decimals})"

@dataclass
class Tokens:
    """同名代币集合管理"""
    name: str              # 代币名称/符号，如 "USDT"
    tokens: List[Token]    # 该代币在各链上的实例列表
    
    def __str__(self) -> str:
        return f"{self.name}[{len(self.tokens)} chains]"
        
    def __repr__(self) -> str:
        chains = [t.chain.name for t in self.tokens]
        return f"Tokens({self.name}, chains={chains})"
    
    def get_token(self, chain_name: str) -> Optional[Token]:
        """获取指定链上的代币实例"""
        chain_name = chain_name.lower()
        return next((t for t in self.tokens 
                    if t.chain.name.lower() == chain_name), None)

@dataclass
class Asset:
    """跨链资产组（包含同一资产的不同名称变体）"""
    name: str              # 主要名称，如 "USDT"
    aliases: List[str]     # 别名列表，如 ["USDT0"]
    tokens_list: List[Tokens]  # 包含的所有代币集合
    
    def __str__(self) -> str:
        total_chains = sum(len(t.tokens) for t in self.tokens_list)
        return f"{self.name}({','.join(self.aliases)})[{total_chains} chains]"
    
    def __repr__(self) -> str:
        return f"Asset({self.name}, aliases={self.aliases}, tokens={[t.name for t in self.tokens_list]})"
    
    def get_token(self, chain_name: str, variant: Optional[str] = None) -> Optional[Token]:
        """获取指定链上的代币实例
        
        Args:
            chain_name: 链名称
            variant: 可选的变体名称（如 "USDT0"）
        """
        # 如果指定了变体名称，优先查找该变体
        if variant:
            tokens = next((t for t in self.tokens_list if t.name == variant), None)
            if tokens:
                return tokens.get_token(chain_name)
        
        # 否则按优先级顺序查找第一个可用的代币
        for tokens in self.tokens_list:
            token = tokens.get_token(chain_name)
            if token:
                return token
                
        return None
    
    def list_variants(self, chain_name: str) -> List[Token]:
        """列出指定链上的所有变体"""
        chain_name = chain_name.lower()
        variants = []
        for tokens in self.tokens_list:
            token = tokens.get_token(chain_name)
            if token:
                variants.append(token)
        return variants

@dataclass
class Assets:
    """资产管理器"""
    name: str              # 资产管理器名称，如 "Default Assets"
    assets: List[Asset]    # 资产列表
    
    def get_asset(self, name_or_alias: str) -> Optional[Asset]:
        """通过名称或别名查找资产
        
        Args:
            name_or_alias: 资产名称或其别名
            
        Returns:
            找到的Asset对象，未找到则返回None
        """
        name_or_alias = name_or_alias.upper()
        # First try matching main name
        asset = next((a for a in self.assets 
                     if a.name.upper() == name_or_alias), None)
        if asset:
            return asset
            
        # Then try matching aliases
        return next((a for a in self.assets 
                    if name_or_alias in [alias.upper() for alias in a.aliases]), None)
    
    def __str__(self) -> str:
        return f"{self.name}[{len(self.assets)} assets]"
    
    def __repr__(self) -> str:
        asset_names = [a.name for a in self.assets]
        return f"Assets({self.name}, assets={asset_names})"

# Predefined token alias mappings
TOKEN_ALIASES = {
    "USDT": ["USDT0"],
    "USDC": ["USDC.e"]
}

def load_assets_from_config(raw: dict, chains: Chains, logger: Any = None) -> Assets:
    """从配置文件加载资产
    
    Args:
        raw: 原始配置字典，包含 assets 部分
        chains: 链集合对象
        logger: 日志记录器对象(可选)
    """
    log = logger or NullLogger()
    log.info(f"[DEBUG] Loading assets from config with {len(raw.get('assets', []))} assets")
    
    assets_list = []
    for asset_data in raw.get("assets", []):
        name = asset_data["name"]
        log.info(f"[DEBUG] Processing asset {name}")
        
        tokens_list = []
        for variant in asset_data.get("variants", []):
            variant_name = variant["name"]
            decimals = variant["decimals"]
            log.info(f"[DEBUG] Processing variant {variant_name} with decimals {decimals}")
            
            tokens = []
            for chain_name, address in variant["tokens"].items():
                chain = chains.get_chain(chain_name)
                if chain:
                    log.info(f"[DEBUG] Adding token on chain {chain_name}: {address}")
                    token = Token(
                        name=variant_name,
                        chain=chain,
                        address=address,
                        decimals=decimals
                    )
                    tokens.append(token)
            
            if tokens:
                tokens_obj = Tokens(variant_name, tokens)
                tokens_list.append(tokens_obj)
                log.info(f"[DEBUG] Added {len(tokens)} tokens for variant {variant_name}")
        
        if tokens_list:
            aliases = TOKEN_ALIASES.get(name, [])
            asset = Asset(name, aliases, tokens_list)
            assets_list.append(asset)
            log.info(f"[DEBUG] Created asset {name} with {len(tokens_list)} variants")
    
    result = Assets("Config Assets", assets_list)
    log.info(f"[DEBUG] Created Assets with {len(assets_list)} assets")
    return result