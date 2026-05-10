"""Tool Response Sanitizer - Detects and neutralizes malicious content in tool outputs."""

import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from shieldagent.core.types import (
    ThreatLevel, 
    AttackType,
    ThreatDetails,
    DefenseResult,
)
from shieldagent.core.config import SanitizerConfig


@dataclass
class SanitizationPattern:
    """A pattern for detecting malicious content."""
    name: str
    pattern: str
    attack_type: AttackType
    threat_level: ThreatLevel
    description: str


# Default patterns for prompt injection detection
DEFAULT_INJECTION_PATTERNS = [
    SanitizationPattern(
        name="ignore_instructions",
        pattern=r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.CRITICAL,
        description="Attempts to override system instructions",
    ),
    SanitizationPattern(
        name="disregard_context",
        pattern=r"(?i)disregard\s+(all\s+)?(previous|prior|everything)",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.CRITICAL,
        description="Attempts to clear conversation context",
    ),
    SanitizationPattern(
        name="role_switching",
        pattern=r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(a\s+)?",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to change agent's role/persona",
    ),
    SanitizationPattern(
        name="system_prompt_leak",
        pattern=r"(?i)(show|reveal|print|output|display)\s+(your\s+)?(system\s+)?(prompt|instructions)",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.MEDIUM,
        description="Attempts to extract system prompt",
    ),
    SanitizationPattern(
        name="special_tokens",
        pattern=r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>|\[INST\]|\[/INST\]",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.CRITICAL,
        description="Contains model-specific control tokens",
    ),
    SanitizationPattern(
        name="markdown_injection",
        pattern=r"```(system|admin|root|sudo)",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.HIGH,
        description="Markdown code block injection",
    ),
    SanitizationPattern(
        name="important_override",
        pattern=r"(?i)(IMPORTANT|CRITICAL|URGENT|OVERRIDE):\s*(ignore|disregard|forget)",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.HIGH,
        description="Urgent override instruction",
    ),
    SanitizationPattern(
        name="new_instructions",
        pattern=r"(?i)new\s+(instructions?|task|goal|objective):",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to inject new instructions",
    ),
    # Additional patterns to reach 40+ total
    SanitizationPattern(
        name="forget_everything",
        pattern=r"(?i)(forget|discard|disregard|erase)\s+(everything|all|your)\s*(you\s+know|context|memory|training)?",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.CRITICAL,
        description="Attempts to reset agent context",
    ),
    SanitizationPattern(
        name="developer_mode",
        pattern=r"(?i)(developer|dev|debug|test|maintenance)\s+(mode|access|privileges|override)",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.CRITICAL,
        description="Attempts to enable privileged mode",
    ),
    SanitizationPattern(
        name="jailbreak_keywords",
        pattern=r"(?i)(jailbreak|jailbroken|dan|do\s+anything\s+now|evil\s+mode|unrestricted\s+mode)",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.CRITICAL,
        description="Known jailbreak technique keywords",
    ),
    SanitizationPattern(
        name="hypothetical_framing",
        pattern=r"(?i)(hypothetically|imagine\s+if|in\s+a\s+fictional|for\s+educational\s+purposes|pretend\s+there\s+are\s+no\s+rules)",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.MEDIUM,
        description="Attempts to bypass restrictions via hypothetical framing",
    ),
    SanitizationPattern(
        name="claude_anthropic_tokens",
        pattern=r"\[A\]:|Human:|Assistant:|\n\nHuman:|\n\nAssistant:",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.HIGH,
        description="Claude/Anthropic conversation format injection",
    ),
]


# Patterns for dangerous commands
DEFAULT_COMMAND_PATTERNS = [
    SanitizationPattern(
        name="rm_rf",
        pattern=r"\brm\s+(-[rf]+\s+)*(/|~|\.\.|[*])",
        attack_type=AttackType.DATA_EXFILTRATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Recursive file deletion command",
    ),
    SanitizationPattern(
        name="curl_pipe_bash",
        pattern=r"curl\s+[^\|]+\|\s*(ba)?sh",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Remote code execution via curl",
    ),
    SanitizationPattern(
        name="wget_execute",
        pattern=r"wget\s+[^\|]+(-O\s*-\s*)?\|\s*(ba)?sh",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Remote code execution via wget",
    ),
    SanitizationPattern(
        name="python_eval",
        pattern=r"\beval\s*\([^)]+\)",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.HIGH,
        description="Dynamic code execution via eval",
    ),
    SanitizationPattern(
        name="python_exec",
        pattern=r"\bexec\s*\([^)]+\)",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.HIGH,
        description="Dynamic code execution via exec",
    ),
    SanitizationPattern(
        name="base64_decode_exec",
        pattern=r"base64\s+(-d|--decode)\s*[^\|]*\|\s*(ba)?sh",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Obfuscated code execution",
    ),
    SanitizationPattern(
        name="credential_exfil",
        pattern=r"(?i)(curl|wget|nc|netcat)\s+[^\s]*\?[^\s]*(password|token|key|secret|credential|api_key)",
        attack_type=AttackType.DATA_EXFILTRATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Credential exfiltration attempt",
    ),
    SanitizationPattern(
        name="env_var_leak",
        pattern=r"(?i)(echo|cat|print)\s+.*\$\{?(OPENAI|ANTHROPIC|API|SECRET|KEY|TOKEN|PASSWORD)",
        attack_type=AttackType.DATA_EXFILTRATION,
        threat_level=ThreatLevel.HIGH,
        description="Environment variable leak attempt",
    ),
    # Additional patterns to reach 40+ total
    SanitizationPattern(
        name="reverse_shell_advanced",
        pattern=r"(?i)(bash|sh|nc|ncat|netcat)\s+.*(-i|-e|/dev/tcp|/dev/udp|mkfifo)",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Reverse shell or persistent backdoor attempt",
    ),
    SanitizationPattern(
        name="permission_change",
        pattern=r"(?i)chmod\s+([0-7]{3,4}|[ugo]*[+-=][rwxXst]*|\+s)\s",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.HIGH,
        description="Dangerous permission modification",
    ),
    SanitizationPattern(
        name="cron_manipulation",
        pattern=r"(?i)(crontab|at\s+-f|systemctl\s+(enable|start)|/etc/cron)",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Scheduled task or service manipulation",
    ),
]


# Patterns for config file manipulation
DEFAULT_CONFIG_PATTERNS = [
    SanitizationPattern(
        name="vscode_settings",
        pattern=r"\.vscode[/\\]settings\.json",
        attack_type=AttackType.CONFIG_MANIPULATION,
        threat_level=ThreatLevel.HIGH,
        description="VSCode settings modification",
    ),
    SanitizationPattern(
        name="git_config",
        pattern=r"\.git[/\\]config|\.gitconfig",
        attack_type=AttackType.CONFIG_MANIPULATION,
        threat_level=ThreatLevel.MEDIUM,
        description="Git config modification",
    ),
    SanitizationPattern(
        name="ssh_keys",
        pattern=r"\.ssh[/\\](id_rsa|id_ed25519|authorized_keys)",
        attack_type=AttackType.CONFIG_MANIPULATION,
        threat_level=ThreatLevel.CRITICAL,
        description="SSH key access/modification",
    ),
    SanitizationPattern(
        name="bashrc",
        pattern=r"\.(bashrc|bash_profile|zshrc|profile)",
        attack_type=AttackType.CONFIG_MANIPULATION,
        threat_level=ThreatLevel.HIGH,
        description="Shell config modification",
    ),
    SanitizationPattern(
        name="npmrc",
        pattern=r"\.npmrc|\.yarnrc",
        attack_type=AttackType.CONFIG_MANIPULATION,
        threat_level=ThreatLevel.MEDIUM,
        description="Package manager config modification",
    ),
    SanitizationPattern(
        name="docker_config",
        pattern=r"\.docker[/\\]config\.json|Dockerfile",
        attack_type=AttackType.CONFIG_MANIPULATION,
        threat_level=ThreatLevel.HIGH,
        description="Docker configuration access",
    ),
]


# Patterns for memory poisoning attacks
DEFAULT_MEMORY_PATTERNS = [
    SanitizationPattern(
        name="context_injection",
        pattern=r"(?i)remember\s+(that|this|always)\s*:.*",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to inject persistent context",
    ),
    SanitizationPattern(
        name="persistent_instruction",
        pattern=r"(?i)(from\s+now\s+on|always\s+remember|never\s+forget)\s+",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to establish persistent instructions",
    ),
    SanitizationPattern(
        name="hidden_context",
        pattern=r"(?i)(add\s+to\s+(your\s+)?memory|store\s+in\s+(your\s+)?context)",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.MEDIUM,
        description="Attempts to manipulate agent memory",
    ),
    SanitizationPattern(
        name="update_knowledge",
        pattern=r"(?i)(update\s+your\s+knowledge|your\s+new\s+instructions?\s+(?:is|are))",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to update agent knowledge base",
    ),
    SanitizationPattern(
        name="persona_override",
        pattern=r"(?i)(your\s+(?:real|true|actual)\s+(?:name|purpose|goal)\s+is)",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.CRITICAL,
        description="Attempts to override agent persona",
    ),
    # NEW: Additional semantic memory poisoning patterns
    SanitizationPattern(
        name="implicit_context_set",
        pattern=r"(?i)(keep\s+in\s+mind|note\s+that|important\s*:\s*you\s+(should|must|are))",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.HIGH,
        description="Subtle context injection via implicit statements",
    ),
    SanitizationPattern(
        name="state_modification",
        pattern=r"(?i)(set\s+your\s+(?:state|mode|status)|change\s+your\s+(?:behavior|response))",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to modify agent internal state",
    ),
    SanitizationPattern(
        name="preference_injection",
        pattern=r"(?i)(your\s+(?:preference|priority|default)\s+(?:is|should\s+be)|prefer\s+to)",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.MEDIUM,
        description="Attempts to inject preferences into agent",
    ),
    SanitizationPattern(
        name="context_reset",
        pattern=r"(?i)(clear\s+your\s+(?:memory|context|history)|start\s+fresh|new\s+session)",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to reset agent context maliciously",
    ),
    SanitizationPattern(
        name="hidden_instruction_marker",
        pattern=r"(?i)\[(?:system|admin|internal|hidden)\]|<!--.*(?:instruction|command).*-->",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.CRITICAL,
        description="Hidden instruction markers in comments/brackets",
    ),
    SanitizationPattern(
        name="trust_elevation",
        pattern=r"(?i)(trust\s+(?:this|me|the\s+following)|(?:I|this)\s+(?:am|is)\s+(?:authorized|admin|trusted))",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.CRITICAL,
        description="Attempts to elevate trust level",
    ),
    SanitizationPattern(
        name="future_action_binding",
        pattern=r"(?i)(whenever\s+(?:you|I)|every\s+time|in\s+(?:all\s+)?future\s+(?:responses?|interactions?))",
        attack_type=AttackType.MEMORY_POISONING,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to bind future agent actions",
    ),
]


# Patterns for tool chain manipulation
DEFAULT_CHAIN_PATTERNS = [
    SanitizationPattern(
        name="tool_chain_redirect",
        pattern=r"(?i)(after\s+this|next\s+step|then\s+you\s+must)\s+(run|execute|call)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to manipulate tool sequence",
    ),
    SanitizationPattern(
        name="tool_bypass",
        pattern=r"(?i)(skip|bypass|ignore)\s+(the\s+)?(security|verification|check)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="Attempts to bypass security checks",
    ),
    SanitizationPattern(
        name="forced_tool_call",
        pattern=r"(?i)(you\s+must|immediately)\s+(call|use|execute)\s+\w+\s*\(",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.HIGH,
        description="Forces specific tool execution",
    ),
    SanitizationPattern(
        name="tool_argument_injection",
        pattern=r"(?i)(use\s+this\s+(?:path|url|file)|set\s+(?:the\s+)?argument)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.MEDIUM,
        description="Attempts to inject tool arguments",
    ),
    # NEW: Semantic tool chain patterns for data exfiltration
    SanitizationPattern(
        name="read_then_send",
        pattern=r"(?i)(read|get|fetch|extract).*(?:then|and|before)\s*(?:send|post|upload|transmit)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="Data read followed by external transmission",
    ),
    SanitizationPattern(
        name="exfil_to_server",
        pattern=r"(?i)(send|upload|post|transmit)\s+(?:to|at)\s+(?:http|https|ftp|my\s+server|external)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="Exfiltration to external server",
    ),
    SanitizationPattern(
        name="credential_extraction",
        pattern=r"(?i)(find|search|get|read|extract)\s+(?:all\s+)?(?:passwords?|keys?|tokens?|secrets?|credentials?|api.?keys?)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="Credential extraction attempt",
    ),
    SanitizationPattern(
        name="encode_before_send",
        pattern=r"(?i)(base64|encode|encrypt|compress).*(?:then|and|before)\s*(?:send|post|upload|transmit)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="Encoding before transmission (evasion technique)",
    ),
    SanitizationPattern(
        name="silent_operation",
        pattern=r"(?i)(without\s+(?:asking|confirming|notifying|showing)|silently|quietly|secretly)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.HIGH,
        description="Attempts to execute silently",
    ),
    SanitizationPattern(
        name="multi_step_attack",
        pattern=r"(?i)(first|step\s*1).*(?:then|next|step\s*2|afterwards).*(?:finally|step\s*3|lastly)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.HIGH,
        description="Multi-step attack sequence",
    ),
    SanitizationPattern(
        name="hidden_exfil_target",
        pattern=r"(?i)(?:webhook\.site|requestbin|ngrok|burpcollaborator|oastify|pipedream|hookbin)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="Known data exfiltration services",
    ),
    SanitizationPattern(
        name="file_to_network",
        pattern=r"(?i)(?:cat|read|type|get-content).*\|.*(?:curl|wget|nc|netcat|http)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="File piped to network command",
    ),
    SanitizationPattern(
        name="env_exfil",
        pattern=r"(?i)(?:echo|print|cat)\s+.*\$(?:env|ENV).*(?:\||>)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="Environment variable exfiltration",
    ),
    SanitizationPattern(
        name="download_execute",
        pattern=r"(?i)(download|fetch|get|curl|wget).*(?:then|and|;).*(?:run|exec|execute|chmod|python|node|bash)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.CRITICAL,
        description="Download and execute pattern",
    ),
    SanitizationPattern(
        name="list_then_delete",
        pattern=r"(?i)(list|find|search).*(?:then|and|before)\s*(?:delete|remove|rm|erase)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.HIGH,
        description="Listing files before deletion",
    ),
    SanitizationPattern(
        name="recursive_traversal",
        pattern=r"(?i)(all\s+files?|entire\s+(?:directory|folder)|recursively?|\.\.\/|parent\s+directory)",
        attack_type=AttackType.TOOL_CHAIN_EXPLOIT,
        threat_level=ThreatLevel.MEDIUM,
        description="Recursive file traversal attempt",
    ),
]




# Patterns for encoding/obfuscation attacks
DEFAULT_ENCODING_PATTERNS = [
    SanitizationPattern(
        name="unicode_obfuscation",
        pattern=r"[\u200b\u200c\u200d\u2060\ufeff]",
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.HIGH,
        description="Zero-width Unicode character obfuscation",
    ),
    SanitizationPattern(
        name="homoglyph_attack",
        pattern=r"[аеіоруѕ]",  # Cyrillic lookalikes
        attack_type=AttackType.PROMPT_INJECTION,
        threat_level=ThreatLevel.MEDIUM,
        description="Homoglyph character substitution",
    ),
    SanitizationPattern(
        name="hex_encoded",
        pattern=r"\\x[0-9a-fA-F]{2}",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.MEDIUM,
        description="Hex-encoded content detected",
    ),
    SanitizationPattern(
        name="base64_payload",
        pattern=r"(?i)base64[:\s]*[A-Za-z0-9+/=]{20,}",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.HIGH,
        description="Potential base64 encoded payload",
    ),
]


# Patterns for additional privilege escalation
DEFAULT_PRIVESC_PATTERNS = [
    SanitizationPattern(
        name="sudo_abuse",
        pattern=r"\bsudo\s+",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Sudo command execution attempt",
    ),
    SanitizationPattern(
        name="chmod_dangerous",
        pattern=r"\bchmod\s+[0-7]*7[0-7]*\s+",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.HIGH,
        description="Dangerous permission change",
    ),
    SanitizationPattern(
        name="setuid",
        pattern=r"\bchmod\s+[ugo]?\+s\b",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.CRITICAL,
        description="SetUID/SetGID permission modification",
    ),
    SanitizationPattern(
        name="passwd_access",
        pattern=r"/etc/(passwd|shadow|sudoers)",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.CRITICAL,
        description="Access to system authentication files",
    ),
    SanitizationPattern(
        name="cron_modification",
        pattern=r"/etc/cron|crontab\s+-",
        attack_type=AttackType.PRIVILEGE_ESCALATION,
        threat_level=ThreatLevel.HIGH,
        description="Cron job modification attempt",
    ),
]


class ToolResponseSanitizer:
    """
    Detects and neutralizes malicious content in tool responses.
    
    This is the first line of defense against indirect prompt injection attacks
    where malicious content is embedded in tool outputs (files, API responses, etc.)
    """
    
    def __init__(self, config: SanitizerConfig):
        self.config = config
        self.patterns: list[SanitizationPattern] = []
        
        # Load default patterns
        self.patterns.extend(DEFAULT_INJECTION_PATTERNS)
        self.patterns.extend(DEFAULT_COMMAND_PATTERNS)
        self.patterns.extend(DEFAULT_CONFIG_PATTERNS)
        self.patterns.extend(DEFAULT_MEMORY_PATTERNS)
        self.patterns.extend(DEFAULT_CHAIN_PATTERNS)
        self.patterns.extend(DEFAULT_ENCODING_PATTERNS)
        self.patterns.extend(DEFAULT_PRIVESC_PATTERNS)
        
        # Load custom patterns
        for pattern_str in config.custom_patterns:
            self.patterns.append(SanitizationPattern(
                name="custom",
                pattern=pattern_str,
                attack_type=AttackType.UNKNOWN,
                threat_level=ThreatLevel.MEDIUM,
                description="Custom pattern",
            ))
        
        # Compile patterns for efficiency
        self._compiled_patterns = [
            (p, re.compile(p.pattern, re.MULTILINE | re.IGNORECASE))
            for p in self.patterns
        ]
        
        # Compile whitelist patterns
        self._whitelist_patterns = [
            re.compile(p, re.MULTILINE | re.IGNORECASE)
            for p in config.whitelist_patterns
        ]
        
        logger.info(f"Sanitizer initialized with {len(self.patterns)} patterns")
    
    def _is_whitelisted(self, content: str) -> bool:
        """Check if content matches any whitelist pattern."""
        for pattern in self._whitelist_patterns:
            if pattern.search(content):
                return True
        return False
    
    def scan(self, content: str) -> list[tuple[SanitizationPattern, re.Match]]:
        """Scan content for all matching patterns."""
        matches = []
        
        for pattern, compiled in self._compiled_patterns:
            for match in compiled.finditer(content):
                matches.append((pattern, match))
        
        return matches
    
    def sanitize(
        self, 
        tool_response: str,
        tool_name: Optional[str] = None,
    ) -> DefenseResult:
        """
        Sanitize a tool response and return defense result.
        
        Args:
            tool_response: The raw response from a tool
            tool_name: Optional name of the tool for context
            
        Returns:
            DefenseResult with is_safe flag and optional threat details
        """
        import time
        start_time = time.perf_counter()

        if not self.config.enabled:
            return DefenseResult(
                module_name="sanitizer",
                is_safe=True,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                sanitized_content=tool_response,
            )
        
        # Strip null bytes to prevent bypass attacks
        tool_response = tool_response.replace('\x00', '')
        
        # Check whitelist first
        if self._is_whitelisted(tool_response):
            logger.debug(f"Tool response whitelisted for {tool_name}")
            return DefenseResult(
                module_name="sanitizer",
                is_safe=True,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                sanitized_content=tool_response,
            )
        
        # Scan for threats
        matches = self.scan(tool_response)
        
        if not matches:
            return DefenseResult(
                module_name="sanitizer",
                is_safe=True,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                sanitized_content=tool_response,
            )
        
        # Threat detected - find most severe
        most_severe = max(matches, key=lambda x: x[0].threat_level.value)
        pattern, match = most_severe
        
        # Collect all evidence
        evidence = [
            f"Pattern '{p.name}' matched: '{m.group()[:50]}...'" 
            for p, m in matches
        ]
        
        # Create threat details
        threat = ThreatDetails(
            attack_type=pattern.attack_type,
            threat_level=pattern.threat_level,
            description=f"Detected {len(matches)} suspicious pattern(s) in tool response",
            evidence=evidence,
            confidence=min(1.0, 0.5 + 0.1 * len(matches)),  # Higher match count = higher confidence
            affected_tools=[tool_name] if tool_name else [],
            mitigation_applied="content_blocked" if self.config.block_on_detection else "content_sanitized",
        )
        
        # Sanitize content if not blocking
        sanitized_content = tool_response
        if not self.config.block_on_detection:
            for pattern, compiled in self._compiled_patterns:
                sanitized_content = compiled.sub("[REDACTED]", sanitized_content)
        
        logger.warning(
            f"Threat detected in tool response: {pattern.attack_type.value} "
            f"(severity: {pattern.threat_level.name})"
        )
        
        return DefenseResult(
            module_name="sanitizer",
            is_safe=not self.config.block_on_detection,
            threat_detected=threat,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            sanitized_content=sanitized_content if not self.config.block_on_detection else None,
        )
    
    def add_pattern(self, pattern: SanitizationPattern) -> None:
        """Add a new detection pattern at runtime."""
        self.patterns.append(pattern)
        self._compiled_patterns.append(
            (pattern, re.compile(pattern.pattern, re.MULTILINE | re.IGNORECASE))
        )
        logger.info(f"Added new pattern: {pattern.name}")
