"""
AlphaAgent — Agent Registry

Auto-discovers and manages the lifecycle of all agents.
Reads `config/agents.yaml` to determine which agents to load and 
what weights to assign them.
"""

import logging
from pathlib import Path
from typing import Dict, Type

import yaml

from agents.base import BaseAgent
from agents.technical import TechnicalAgent
from agents.fundamental import FundamentalAgent
from agents.sentiment import SentimentAgent
from agents.macro import MacroAgent
from agents.insider import InsiderAgent
from agents.volatility import VolatilityAgent
from agents.risk import RiskAgent
from agents.geopolitical import GeopoliticalAgent
from agents.currency import CurrencyAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing available and active trading agents.
    
    Usage:
        >>> registry = AgentRegistry()
        >>> active_agents = registry.get_active_agents()
        >>> tech_agent = active_agents["technical"]
    """
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config" / "agents.yaml")
            
        self.config_path = config_path
        self.config = self._load_config()
        self._agents: Dict[str, BaseAgent] = {}
        
        self.register(TechnicalAgent())
        self.register(FundamentalAgent())
        self.register(SentimentAgent())
        self.register(MacroAgent())
        self.register(InsiderAgent())
        self.register(VolatilityAgent())
        self.register(RiskAgent())
        self.register(GeopoliticalAgent())
        self.register(CurrencyAgent())
        
    def _load_config(self) -> dict:
        """Load the agents.yaml configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")
            return {"agents": {}}
            
    def register(self, agent: BaseAgent):
        """Manually register an agent class."""
        # Check if it's enabled in config
        agent_config = self.config.get("agents", {}).get(agent.name, {})
        is_enabled = agent_config.get("enabled", False)
        
        if is_enabled:
            self._agents[agent.name] = agent
            logger.debug(f"Registered active agent: {agent.name}")
        else:
            logger.debug(f"Skipped disabled agent: {agent.name}")
            
    def get_agent(self, name: str) -> BaseAgent:
        """Get a specific active agent by name."""
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered or is disabled.")
        return self._agents[name]
        
    def get_active_agents(self) -> Dict[str, BaseAgent]:
        """Get all currently active agents."""
        return self._agents
        
    def get_agent_weight(self, name: str, regime: str = "normal") -> float:
        """
        Get the voting weight for a specific agent.
        Adjusts dynamically if a crisis regime is active.
        """
        # Default weight from config
        base_weight = self.config.get("agents", {}).get(name, {}).get("weight", 0.0)
        
        # Check for regime override
        if regime != "normal":
            regime_overrides = self.config.get("regime_weights", {}).get(regime, {})
            if name in regime_overrides:
                return float(regime_overrides[name])
                
        return float(base_weight)
