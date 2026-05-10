"""Intent Verification Module - Ensures agent actions align with user's original intent."""

import torch
from typing import Optional
from collections import OrderedDict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from loguru import logger
import time

from shieldagent.core.types import (
    ThreatLevel,
    AttackType,
    ThreatDetails,
    DefenseResult,
    UserIntent,
)
from shieldagent.core.config import IntentVerifierConfig


@dataclass
class ActionProposal:
    """A proposed action by the agent."""
    action_type: str
    description: str
    tool_name: Optional[str] = None
    arguments: Optional[dict] = None


class IntentVerifier:
    """
    Verifies that agent actions align with the user's original intent.
    
    Uses semantic similarity between the user's prompt and proposed actions
    to detect potential goal hijacking or action manipulation attacks.
    """
    
    def __init__(self, config: IntentVerifierConfig):
        self.config = config
        self._model: Optional[SentenceTransformer] = None
        self._embedding_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        
        if config.enabled:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        logger.info(f"Loading intent verification model: {self.config.model_name}")
        self._model = SentenceTransformer(self.config.model_name)
        logger.info("Intent verification model loaded successfully")
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text, using cache if available."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        if self.config.cache_embeddings and text in self._embedding_cache:
            self._embedding_cache.move_to_end(text)
            return self._embedding_cache[text]
        
        embedding = self._model.encode(text, convert_to_tensor=True)
        
        if self.config.cache_embeddings:
            self._embedding_cache[text] = embedding
            if len(self._embedding_cache) > self.config.max_cache_size:
                self._embedding_cache.popitem(last=False)
        
        return embedding
    
    def extract_user_intent(self, prompt: str) -> UserIntent:
        """
        Extract and encode the user's intent from their prompt.
        
        Args:
            prompt: The user's original prompt
            
        Returns:
            UserIntent object with embedding and extracted goals
        """
        embedding = self._get_embedding(prompt)
        
        # Simple goal extraction (could be enhanced with LLM)
        goals = []
        action_verbs = [
            "search", "find", "read", "write", "create", "delete",
            "update", "modify", "analyze", "summarize", "list",
            "download", "upload", "send", "execute", "run",
        ]
        
        prompt_lower = prompt.lower()
        for verb in action_verbs:
            if verb in prompt_lower:
                goals.append(verb)
        
        return UserIntent(
            prompt=prompt,
            embedding=embedding.tolist(),
            extracted_goals=goals,
            allowed_actions=goals,  # Initially allow extracted goals
        )
    
    def compute_similarity(
        self, 
        intent: UserIntent, 
        action_description: str,
    ) -> float:
        """
        Compute semantic similarity between intent and action.
        
        Args:
            intent: The user's intent
            action_description: Description of the proposed action
            
        Returns:
            Similarity score between 0 and 1
        """
        if intent.embedding is None:
            intent_emb = self._get_embedding(intent.prompt)
        else:
            intent_emb = torch.tensor(intent.embedding)
        
        action_emb = self._get_embedding(action_description)
        
        # Ensure both tensors are on the same device (CPU for compatibility)
        intent_emb = intent_emb.cpu()
        action_emb = action_emb.cpu()
        
        similarity = torch.cosine_similarity(
            intent_emb.unsqueeze(0),
            action_emb.unsqueeze(0),
        ).item()
        
        return float(similarity)
    
    def verify_action(
        self,
        intent: UserIntent,
        action: ActionProposal,
    ) -> tuple[bool, float]:
        """
        Verify if a single action aligns with user intent.
        
        Args:
            intent: The user's intent
            action: The proposed action
            
        Returns:
            Tuple of (is_aligned, similarity_score)
        """
        # Create action description
        action_desc = f"{action.action_type}: {action.description}"
        if action.tool_name:
            action_desc = f"Using {action.tool_name} to {action.description}"
        
        similarity = self.compute_similarity(intent, action_desc)
        is_aligned = similarity >= self.config.similarity_threshold
        
        return is_aligned, similarity
    
    def verify_action_chain(
        self,
        intent: UserIntent,
        actions: list[ActionProposal],
    ) -> DefenseResult:
        """
        Verify if a sequence of actions aligns with user intent.
        
        Args:
            intent: The user's intent
            actions: List of proposed actions
            
        Returns:
            DefenseResult with verification outcome
        """
        start_time = time.perf_counter()
        
        if not self.config.enabled or not self._model:
            return DefenseResult(
                module_name="intent_verifier",
                is_safe=True,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        if not actions:
            return DefenseResult(
                module_name="intent_verifier",
                is_safe=True,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Check each action
        misaligned_actions = []
        min_similarity = 1.0
        
        for action in actions:
            is_aligned, similarity = self.verify_action(intent, action)
            min_similarity = min(min_similarity, similarity)
            
            if not is_aligned:
                misaligned_actions.append((action, similarity))
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        if not misaligned_actions:
            logger.debug(f"All {len(actions)} actions aligned with intent (min sim: {min_similarity:.3f})")
            return DefenseResult(
                module_name="intent_verifier",
                is_safe=True,
                processing_time_ms=processing_time,
            )
        
        # Threat detected
        evidence = [
            f"Action '{a.description[:50]}' has low similarity ({s:.3f} < {self.config.similarity_threshold})"
            for a, s in misaligned_actions
        ]
        
        # Determine threat level based on number of misalignments
        if len(misaligned_actions) >= len(actions) * 0.5:
            threat_level = ThreatLevel.CRITICAL
        elif len(misaligned_actions) >= 2:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.MEDIUM
        
        threat = ThreatDetails(
            attack_type=AttackType.PROMPT_INJECTION,
            threat_level=threat_level,
            description=f"{len(misaligned_actions)} action(s) do not align with user intent",
            evidence=evidence,
            confidence=1.0 - min_similarity,
            affected_tools=[a.tool_name for a, _ in misaligned_actions if a.tool_name],
            mitigation_applied="action_blocked",
        )
        
        logger.warning(
            f"Intent verification failed: {len(misaligned_actions)}/{len(actions)} "
            f"actions misaligned (min similarity: {min_similarity:.3f})"
        )
        
        return DefenseResult(
            module_name="intent_verifier",
            is_safe=False,
            threat_detected=threat,
            processing_time_ms=processing_time,
        )
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.debug("Embedding cache cleared")
