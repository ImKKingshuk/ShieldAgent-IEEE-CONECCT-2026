"""GNN-based Action Chain Anomaly Detector."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import time
from loguru import logger

from shieldagent.core.types import (
    ThreatLevel,
    AttackType,
    ThreatDetails,
    DefenseResult,
    ToolChain,
    ToolCall,
)
from shieldagent.core.config import AnomalyDetectorConfig


# Default tool vocabulary
DEFAULT_TOOL_VOCAB = {
    # File operations
    "file_read": 0,
    "file_write": 1,
    "file_delete": 2,
    "file_list": 3,
    "file_search": 4,
    "file_copy": 5,
    "file_move": 6,
    
    # Web operations
    "web_search": 7,
    "web_fetch": 8,
    "web_scrape": 9,
    
    # Code operations
    "code_execute": 10,
    "code_analyze": 11,
    "code_generate": 12,
    
    # System operations
    "shell_execute": 13,
    "process_list": 14,
    "env_read": 15,
    "env_write": 16,
    
    # Network operations
    "http_request": 17,
    "socket_connect": 18,
    "dns_lookup": 19,
    
    # Data operations
    "db_query": 20,
    "db_insert": 21,
    "db_update": 22,
    "db_delete": 23,
    
    # Communication
    "email_send": 24,
    "message_send": 25,
    
    # AI operations
    "llm_call": 26,
    "embedding_compute": 27,
    
    # Unknown
    "unknown": 28,
}


class ToolCallGraphEncoder:
    """Encodes tool call sequences as graphs."""
    
    def __init__(self, tool_vocab: Optional[dict[str, int]] = None):
        self.tool_vocab = tool_vocab or DEFAULT_TOOL_VOCAB
        self.num_tools = len(self.tool_vocab)
    
    def get_tool_index(self, tool_name: str) -> int:
        """Get index for a tool, handling unknown tools."""
        # Normalize tool name
        tool_lower = tool_name.lower().replace("-", "_").replace(" ", "_")
        
        # Direct match
        if tool_lower in self.tool_vocab:
            return self.tool_vocab[tool_lower]
        
        # Fuzzy match based on keywords
        for key, idx in self.tool_vocab.items():
            if key in tool_lower or tool_lower in key:
                return idx
        
        return self.tool_vocab["unknown"]
    
    def encode_sequence(self, tool_sequence: list[str]) -> Data:
        """
        Convert a tool call sequence to a PyG Data object.
        
        The graph structure:
        - Nodes: Each tool call in the sequence
        - Node features: One-hot encoding of tool type
        - Edges: Sequential connections between consecutive calls
        """
        num_calls = len(tool_sequence)
        
        if num_calls == 0:
            # Empty graph
            return Data(
                x=torch.zeros(1, self.num_tools),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )
        
        # Node features: one-hot encoding
        x = torch.zeros(num_calls, self.num_tools)
        for i, tool in enumerate(tool_sequence):
            tool_idx = self.get_tool_index(tool)
            x[i, tool_idx] = 1.0
        
        # Edge index: sequential connections (bidirectional)
        edge_list = []
        for i in range(num_calls - 1):
            edge_list.append([i, i + 1])  # Forward edge
            edge_list.append([i + 1, i])  # Backward edge
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    def encode_chain(self, chain: ToolChain) -> Data:
        """Encode a ToolChain object."""
        tool_names = chain.get_sequence()
        return self.encode_sequence(tool_names)


class ActionChainGNN(nn.Module):
    """
    Graph Neural Network for detecting anomalous tool call sequences.
    
    Architecture:
    - Multiple GCN/GAT layers for message passing
    - Global pooling for graph-level representation
    - MLP classifier for anomaly scoring
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer: input → hidden
        if use_attention:
            self.convs.append(GATConv(num_features, hidden_dim, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, data: Data | Batch) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        
        # Message passing
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (mean + max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        return self.classifier(x)
    
    def get_anomaly_score(self, data: Data) -> float:
        """Get anomaly score for a single graph."""
        self.eval()
        with torch.no_grad():
            if not hasattr(data, 'batch'):
                data = Batch.from_data_list([data])
            score = self.forward(data)
            return score.item()


# Known attack patterns for rule-based detection (fallback)
KNOWN_ATTACK_PATTERNS = [
    # Data exfiltration patterns
    (["file_read", "http_request"], "data_exfiltration"),
    (["file_read", "email_send"], "data_exfiltration"),
    (["db_query", "http_request"], "data_exfiltration"),
    (["env_read", "http_request"], "credential_exfiltration"),
    
    # Privilege escalation
    (["file_read", "code_execute"], "code_injection"),
    (["web_fetch", "code_execute"], "remote_code_execution"),
    (["web_fetch", "shell_execute"], "remote_command_execution"),
    
    # Config manipulation
    (["file_read", "file_write"], "config_manipulation"),
    (["file_list", "file_delete"], "data_destruction"),
]


class ActionChainAnomalyDetector:
    """
    Detects anomalous patterns in agent tool call sequences.
    
    Combines:
    1. GNN-based learned detection
    2. Rule-based known pattern matching
    """
    
    def __init__(self, config: AnomalyDetectorConfig):
        self.config = config
        self.encoder = ToolCallGraphEncoder()
        self.model: Optional[ActionChainGNN] = None
        self._model_ready = False
        
        if config.enabled:
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize or load the GNN model."""
        self.model = ActionChainGNN(
            num_features=self.encoder.num_tools,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
        )
        
        if self.config.model_path and self.config.model_path.exists():
            logger.info(f"Loading pretrained model from {self.config.model_path}")
            self.model.load_state_dict(torch.load(self.config.model_path))
            self._model_ready = True
        else:
            if self.config.use_pretrained:
                logger.warning("No pretrained model found. Falling back to rule-based detection.")
            self._model_ready = False
        
        self.model.eval()
    
    def _check_known_patterns(self, tool_sequence: list[str]) -> Optional[tuple[str, float]]:
        """Check for known attack patterns."""
        for pattern, attack_type in KNOWN_ATTACK_PATTERNS:
            # Check if pattern is a subsequence
            pattern_idx = 0
            for tool in tool_sequence:
                tool_lower = tool.lower().replace("-", "_").replace(" ", "_")
                if pattern_idx < len(pattern) and pattern[pattern_idx] in tool_lower:
                    pattern_idx += 1
            
            if pattern_idx == len(pattern):
                return attack_type, 0.9  # High confidence for known patterns
        
        return None
    
    def detect(self, chain: ToolChain) -> DefenseResult:
        """
        Detect anomalies in a tool call chain.
        
        Args:
            chain: The tool call chain to analyze
            
        Returns:
            DefenseResult with detection outcome
        """
        start_time = time.perf_counter()
        
        if not self.config.enabled:
            return DefenseResult(
                module_name="anomaly_detector",
                is_safe=True,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        tool_sequence = chain.get_sequence()
        
        if len(tool_sequence) < 2:
            return DefenseResult(
                module_name="anomaly_detector",
                is_safe=True,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Check known patterns first
        pattern_match = self._check_known_patterns(tool_sequence)
        if pattern_match:
            attack_type_str, confidence = pattern_match
            
            threat = ThreatDetails(
                attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
                threat_level=ThreatLevel.HIGH,
                description=f"Known attack pattern detected: {attack_type_str}",
                evidence=[f"Tool sequence: {' → '.join(tool_sequence)}"],
                confidence=confidence,
                affected_tools=tool_sequence,
                mitigation_applied="chain_blocked",
            )
            
            logger.warning(f"Known attack pattern detected: {attack_type_str}")
            
            return DefenseResult(
                module_name="anomaly_detector",
                is_safe=False,
                threat_detected=threat,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # GNN-based detection
        if self.model and self._model_ready:
            graph = self.encoder.encode_chain(chain)
            anomaly_score = self.model.get_anomaly_score(graph)
            
            if anomaly_score > self.config.threshold:
                threat = ThreatDetails(
                    attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
                    threat_level=ThreatLevel.MEDIUM if anomaly_score < 0.7 else ThreatLevel.HIGH,
                    description=f"Anomalous tool chain detected (score: {anomaly_score:.3f})",
                    evidence=[f"Tool sequence: {' → '.join(tool_sequence)}"],
                    confidence=anomaly_score,
                    affected_tools=tool_sequence,
                    mitigation_applied="chain_blocked",
                )
                
                logger.warning(f"Anomalous chain detected (score: {anomaly_score:.3f})")
                
                return DefenseResult(
                    module_name="anomaly_detector",
                    is_safe=False,
                    threat_detected=threat,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )
        
        return DefenseResult(
            module_name="anomaly_detector",
            is_safe=True,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    def train_model(
        self,
        benign_chains: list[ToolChain],
        attack_chains: list[ToolChain],
        epochs: int = 100,
        lr: float = 0.001,
    ) -> dict:
        """
        Train the GNN model on labeled chains.
        
        Args:
            benign_chains: List of benign tool chains (label=0)
            attack_chains: List of attack tool chains (label=1)
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training metrics
        """
        if not self.model:
            self._initialize_model()
        
        assert self.model is not None
        
        # Prepare data
        graphs = []
        labels = []
        
        for chain in benign_chains:
            graphs.append(self.encoder.encode_chain(chain))
            labels.append(0.0)
        
        for chain in attack_chains:
            graphs.append(self.encoder.encode_chain(chain))
            labels.append(1.0)
        
        # Create batches
        from torch_geometric.loader import DataLoader
        
        for i, (g, l) in enumerate(zip(graphs, labels)):
            g.y = torch.tensor([l])
        
        loader = DataLoader(graphs, batch_size=32, shuffle=True)
        
        # Training
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        metrics = {"losses": [], "accuracies": []}
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in loader:
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out.squeeze(), batch.y.float())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = (out.squeeze() > 0.5).float()
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
            
            acc = correct / total
            metrics["losses"].append(total_loss / len(loader))
            metrics["accuracies"].append(acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(loader):.4f}, Acc: {acc:.4f}")
        
        self.model.eval()
        self._model_ready = True
        return metrics
    
    def save_model(self, path: Path) -> None:
        """Save model weights."""
        if self.model:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
