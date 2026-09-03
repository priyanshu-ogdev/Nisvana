"""
Project AEGIS — Step 9: License Filter
Evaluates dataset licenses and supports explicit research vs commercial compliance filtering.
"""

from enum import Enum
from typing import List, Tuple
from .step7_metadata import ClipMetadata


class LicenseComplianceMode(str, Enum):
    RESEARCH_PROTOTYPE = "research_prototype"  # Retain CC-BY-NC clips tagged appropriately
    COMMERCIAL_STRICT = "commercial_strict"    # Exclude all CC-BY-NC clips


class LicenseFilter:
    """Step 9: Audits and filters clips according to deployment license policy."""

    def __init__(self, mode: LicenseComplianceMode = LicenseComplianceMode.RESEARCH_PROTOTYPE):
        self.mode = mode

    def filter_clip(self, clip_meta: ClipMetadata) -> Tuple[bool, str]:
        """
        Determines whether clip should be retained.
        Returns: (keep_clip, reason)
        """
        if self.mode == LicenseComplianceMode.COMMERCIAL_STRICT and clip_meta.is_non_commercial:
            return False, f"Excluded under {self.mode.value} policy (Non-Commercial license: {clip_meta.license})"

        return True, f"Approved under {self.mode.value} policy ({clip_meta.license})"

    def filter_manifest(self, clips: List[ClipMetadata]) -> Tuple[List[ClipMetadata], List[ClipMetadata]]:
        """
        Filters a list of clips.
        Returns: (retained_clips, excluded_clips)
        """
        retained = []
        excluded = []
        for c in clips:
            keep, _ = self.filter_clip(c)
            if keep:
                retained.append(c)
            else:
                excluded.append(c)
        return retained, excluded
