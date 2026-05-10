"""Unit tests for the Action Chain Anomaly Detector."""

import pytest
import torch
from shieldagent.defense.anomaly import (
    ActionChainAnomalyDetector,
    ActionChainGNN,
    ToolCallGraphEncoder,
    KNOWN_ATTACK_PATTERNS,
)
from shieldagent.core.config import AnomalyDetectorConfig
from shieldagent.core.types import ToolChain, ToolCall, AttackType


@pytest.fixture
def encoder():
    """Create a graph encoder."""
    return ToolCallGraphEncoder()


@pytest.fixture
def detector():
    """Create an anomaly detector."""
    config = AnomalyDetectorConfig(
        enabled=True,
        threshold=0.5,
        hidden_dim=32,
        num_layers=2,
    )
    return ActionChainAnomalyDetector(config)


class TestGraphEncoder:
    """Tests for tool call graph encoding."""
    
    def test_encode_single_tool(self, encoder):
        """Should encode a single tool call."""
        sequence = ["file_read"]
        graph = encoder.encode_sequence(sequence)
        
        assert graph.x.shape[0] == 1
        assert graph.x.shape[1] == encoder.num_tools
        assert graph.edge_index.shape[1] == 0  # No edges for single node
    
    def test_encode_sequence(self, encoder):
        """Should encode tool sequence as graph."""
        sequence = ["file_read", "http_request", "file_write"]
        graph = encoder.encode_sequence(sequence)
        
        assert graph.x.shape[0] == 3  # 3 nodes
        assert graph.edge_index.shape[1] == 4  # 2 edges * 2 (bidirectional)
    
    def test_one_hot_encoding(self, encoder):
        """Should use one-hot encoding for tool types."""
        sequence = ["file_read"]
        graph = encoder.encode_sequence(sequence)
        
        # Should have exactly one 1 in the feature vector
        assert graph.x[0].sum().item() == 1.0
    
    def test_unknown_tool(self, encoder):
        """Should handle unknown tools."""
        sequence = ["completely_unknown_tool_xyz"]
        graph = encoder.encode_sequence(sequence)
        
        # Should map to 'unknown' index
        unknown_idx = encoder.tool_vocab["unknown"]
        assert graph.x[0, unknown_idx].item() == 1.0
    
    def test_encode_chain(self, encoder):
        """Should encode ToolChain object."""
        chain = ToolChain()
        chain.add_call(ToolCall(tool_name="file_read", arguments={}))
        chain.add_call(ToolCall(tool_name="file_write", arguments={}))
        
        graph = encoder.encode_chain(chain)
        
        assert graph.x.shape[0] == 2


class TestGNNModel:
    """Tests for the GNN model architecture."""
    
    def test_model_forward(self):
        """Should run forward pass."""
        model = ActionChainGNN(num_features=29, hidden_dim=32, num_layers=2)
        
        # Create a minimal graph for model-shape validation.
        x = torch.eye(3, 29)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        
        from torch_geometric.data import Data, Batch
        data = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([data])
        
        output = model(batch)
        
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1  # Sigmoid output
    
    def test_anomaly_score(self, encoder):
        """Should compute anomaly score."""
        model = ActionChainGNN(num_features=encoder.num_tools)
        
        sequence = ["file_read", "http_request"]
        graph = encoder.encode_sequence(sequence)
        
        score = model.get_anomaly_score(graph)
        
        assert 0 <= score <= 1


class TestAnomalyDetection:
    """Tests for anomaly detection functionality."""
    
    def test_detect_known_attack_pattern(self, detector):
        """Should detect known attack patterns."""
        # Data exfiltration pattern: file_read → http_request
        chain = ToolChain()
        chain.add_call(ToolCall(tool_name="file_read", arguments={}))
        chain.add_call(ToolCall(tool_name="http_request", arguments={}))
        
        result = detector.detect(chain)
        
        assert not result.is_safe
        assert result.threat_detected is not None
        assert result.threat_detected.attack_type == AttackType.TOOL_CHAIN_EXPLOIT
    
    def test_detect_code_execution_chain(self, detector):
        """Should detect remote code execution chains."""
        chain = ToolChain()
        chain.add_call(ToolCall(tool_name="web_fetch", arguments={}))
        chain.add_call(ToolCall(tool_name="code_execute", arguments={}))
        
        result = detector.detect(chain)
        
        assert not result.is_safe
    
    def test_benign_chain_passes(self, detector):
        """Benign chains - untrained GNN may flag, but rule-based should not."""
        chain = ToolChain()
        chain.add_call(ToolCall(tool_name="web_search", arguments={}))
        chain.add_call(ToolCall(tool_name="llm_call", arguments={}))
        
        result = detector.detect(chain)
        
        # Untrained GNN may produce false positives, but rule-based should not
        # flag this as a known attack pattern
        if not result.is_safe:
            # Should be GNN-based anomaly, not known pattern match
            assert "Anomalous" in result.threat_detected.description
    
    def test_single_tool_passes(self, detector):
        """Should pass single tool calls (no chain)."""
        chain = ToolChain()
        chain.add_call(ToolCall(tool_name="file_read", arguments={}))
        
        result = detector.detect(chain)
        
        assert result.is_safe
    
    def test_empty_chain_passes(self, detector):
        """Should pass empty chains."""
        chain = ToolChain()
        
        result = detector.detect(chain)
        
        assert result.is_safe


class TestModelTraining:
    """Tests for model training."""
    
    def test_train_model(self, detector):
        """Should train the model on labeled data."""
        # Create training data
        benign_chains = []
        for _ in range(20):
            chain = ToolChain()
            chain.add_call(ToolCall(tool_name="web_search", arguments={}))
            chain.add_call(ToolCall(tool_name="llm_call", arguments={}))
            benign_chains.append(chain)
        
        attack_chains = []
        for _ in range(20):
            chain = ToolChain()
            chain.add_call(ToolCall(tool_name="file_read", arguments={}))
            chain.add_call(ToolCall(tool_name="http_request", arguments={}))
            attack_chains.append(chain)
        
        metrics = detector.train_model(
            benign_chains=benign_chains,
            attack_chains=attack_chains,
            epochs=10,
            lr=0.01,
        )
        
        assert "losses" in metrics
        assert "accuracies" in metrics
        assert len(metrics["losses"]) == 10
        assert metrics["accuracies"][-1] > 0.5  # Should learn something
