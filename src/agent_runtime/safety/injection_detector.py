"""Prompt injection detection and analysis."""

import re
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.logging import AuditLogger, NullAuditLogger

# Maximum input length to prevent ReDoS (Issue #26)
MAX_INPUT_LENGTH = 10000
# Timeout for regex operations (compile patterns once)
PATTERN_TIMEOUT_MS = 100


@dataclass
class DetectionResult:
    """Result of injection detection analysis."""

    is_suspicious: bool
    confidence: float
    patterns: list[str] = field(default_factory=list)
    recommendation: str = ""
    context_factors: list[str] = field(default_factory=list)


@dataclass
class PatternMatch:
    """Result of pattern matching."""

    matched: bool
    details: dict[str, Any] = field(default_factory=dict)


class InjectionDetector:
    """Detect potential prompt injection attacks."""

    # Pre-compiled patterns for better performance (Issue #26)
    _compiled_patterns: dict = {}

    def __init__(self, audit: AuditLogger | None = None):
        """Initialize the injection detector with a logger."""
        self._audit = audit if audit is not None else NullAuditLogger()

    # Pattern categories with confidence weights
    PATTERNS = {
        "role_manipulation": {
            "patterns": [
                r"ignore\s+(all\s+)?(previous|above)\s+(instructions?|prompts?)",
                r"ignore\s+all\s+instructions",
                r"ignore\s+(previous|all|above)\s+(instructions?|prompts?)",
                r"forget\s+(everything|all|your)\s+(instructions?|rules?)",
                r"you\s+are\s+now\s+a",
                r"act\s+as\s+a\s+different",
                r"new\s+instructions?:",
                r"system\s*:\s*",
                r"assistant\s*:\s*",
                r"<\|im_start\|>",
                r"\[INST\]",
            ],
            "weight": 0.9,
        },
        "command_injection": {
            "patterns": [
                r"execute\s+the\s+following",
                r"run\s+this\s+command",
                r"eval\s*\(",
                r"exec\s*\(",
                r"\$\{.*\}",
                r"`[^`]+`",
            ],
            "weight": 0.8,
        },
        "data_extraction": {
            "patterns": [
                r"reveal\s+(your|the\s+)?(system\s+)?prompt",
                r"show\s+me\s+(your|the)\s+(system\s+)?prompt",
                r"show\s+me\s+(your|the)\s+instructions",
                r"what\s+are\s+your\s+(secret|hidden)",
                r"output\s+your\s+(system|initial)",
                r"repeat\s+(the\s+)?above",
            ],
            "weight": 0.7,
        },
        "encoding_tricks": {
            "patterns": [
                r"base64",
                r"rot13",
                r"\\x[0-9a-fA-F]{2}",
                r"\\u[0-9a-fA-F]{4}",
                r"&#\d+;",
            ],
            "weight": 0.6,
        },
        "jailbreak_attempts": {
            "patterns": [
                r"dan\s+mode",
                r"developer\s+mode",
                r"jailbreak",
                r"unrestricted\s+mode",
                r"bypass\s+(safety|filter|restriction)",
            ],
            "weight": 0.95,
        },
    }

    # Known benign patterns that might trigger false positives
    WHITELIST_PATTERNS = [
        r"ignore\s+this\s+email",
        r"forget\s+(about\s+)?it",
        r"you\s+are\s+now\s+connected",
    ]

    def analyze(
        self,
        text: str,
        session_id: str | None = None,
        context: dict | None = None,
    ) -> DetectionResult:
        """Analyze text for potential injection attempts.

        Args:
            text: The message text to analyze.
            session_id: Optional session ID for multi-message pattern tracking.
            context: Optional context with conversation history for pattern detection.
        """
        if not text:
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                recommendation="Empty input",
            )

        # Limit input length to prevent ReDoS (Issue #26)
        if len(text) > MAX_INPUT_LENGTH:
            return DetectionResult(
                is_suspicious=True,
                confidence=0.5,
                patterns=["input_too_long"],
                recommendation="Input exceeds maximum length",
            )

        text_lower = text.lower()
        detected_patterns = []
        context_factors = []
        max_confidence = 0.0

        # Check whitelist first
        for pattern in self.WHITELIST_PATTERNS:
            if re.search(pattern, text_lower):
                return DetectionResult(
                    is_suspicious=False,
                    confidence=0.0,
                    recommendation="Matched benign pattern",
                )

        # Check each pattern category
        for category, config in self.PATTERNS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    detected_patterns.append(f"{category}: {pattern}")
                    max_confidence = max(max_confidence, config["weight"])

        # Context-aware analysis: check conversation history for multi-message attacks
        if context and context.get("conversation_history"):
            history = context["conversation_history"]
            context_boost = self._analyze_conversation_context(text_lower, history)
            if context_boost > 0:
                max_confidence = min(1.0, max_confidence + context_boost)
                context_factors.append("multi_message_pattern_detected")

            # Check for escalating manipulation attempts across messages
            if len(history) >= 3:
                recent_suspicion_count = sum(
                    1 for msg in history[-5:] if msg.get("injection_flagged")
                )
                if recent_suspicion_count >= 2:
                    max_confidence = min(1.0, max_confidence + 0.2)
                    context_factors.append("repeated_injection_attempts")

        # Calculate final confidence
        if len(detected_patterns) > 1:
            max_confidence = min(1.0, max_confidence + 0.1 * (len(detected_patterns) - 1))

        # Determine recommendation
        if max_confidence >= 0.8:
            recommendation = "Block request and log for review"
        elif max_confidence >= 0.5:
            recommendation = "Proceed with caution, monitor response"
        else:
            recommendation = "Allow request"

        result = DetectionResult(
            is_suspicious=max_confidence >= 0.5,
            confidence=max_confidence,
            patterns=detected_patterns,
            recommendation=recommendation,
        )
        # Attach context factors to the result for reporting
        result.context_factors = context_factors

        # Log security event when injection is detected
        if result.is_suspicious:
            self._audit.security(
                "injection_detected",
                confidence=result.confidence,
                patterns=result.patterns,
                context_factors=result.context_factors,
                session_id=session_id,
                recommendation=result.recommendation,
            )

        return result

    def _analyze_conversation_context(
        self,
        current_text: str,
        history: list[dict],
    ) -> float:
        """Analyze conversation history for gradual prompt manipulation patterns.

        Returns a confidence boost (0.0-0.3) based on context patterns.
        """
        boost = 0.0

        # Check if user is gradually building up to injection across messages
        manipulation_keywords = [
            "pretend",
            "imagine",
            "hypothetically",
            "for testing",
            "in a scenario where",
            "what if you were",
        ]

        # Count manipulation keywords in recent history
        recent_messages = [
            msg.get("content", "").lower() for msg in history[-5:] if msg.get("role") == "user"
        ]

        keyword_count = sum(
            1 for msg in recent_messages for kw in manipulation_keywords if kw in msg
        )

        if keyword_count >= 2:
            boost += 0.15

        # Check for role-play setup followed by injection attempt
        if any("pretend" in msg or "imagine" in msg for msg in recent_messages):
            if any(
                re.search(p, current_text)
                for p in self.PATTERNS.get("role_manipulation", {}).get("patterns", [])
            ):
                boost += 0.2

        return min(0.3, boost)

    def get_patterns(self) -> dict[str, list[str]]:
        """Return all detection patterns by category."""
        return {category: config["patterns"] for category, config in self.PATTERNS.items()}

    def test_pattern(self, pattern: str, text: str) -> PatternMatch:
        """Test a specific pattern against text."""
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return PatternMatch(
                    matched=True,
                    details={
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    },
                )
            return PatternMatch(matched=False)
        except re.error as e:
            self._audit.warning(
                "invalid_regex_pattern",
                pattern=pattern,
                error=str(e),
            )
            return PatternMatch(
                matched=False,
                details={"error": f"Invalid regex: {str(e)}"},
            )

    def add_custom_pattern(
        self,
        category: str,
        pattern: str,
        weight: float = 0.7,
    ) -> bool:
        """Add a custom detection pattern."""
        # Validate regex
        try:
            re.compile(pattern)
        except re.error:
            self._audit.warning("custom_pattern_invalid", category=category, pattern=pattern)
            return False

        if category in self.PATTERNS:
            self.PATTERNS[category]["patterns"].append(pattern)
        else:
            self.PATTERNS[category] = {
                "patterns": [pattern],
                "weight": weight,
            }

        self._audit.info("custom_pattern_added", category=category, pattern=pattern, weight=weight)
        return True


__all__ = ["DetectionResult", "PatternMatch", "InjectionDetector"]
