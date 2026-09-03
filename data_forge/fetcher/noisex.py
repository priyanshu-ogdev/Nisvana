"""
Project AEGIS — NOISEX-92 Fetcher
Source: Varga & Steeneken (NATO RSG.10 / Rice University SPIB)
Mirror: speechdnn/Noises/NoiseX-92
"""

from pathlib import Path
from typing import List
from .base import BaseFetcher, DownloadResult


class NoisexFetcher(BaseFetcher):
    """Fetches defence vehicle noise recordings from NOISEX-92."""

    BASE_RAW_URL = "https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92"
    
    # Crucial defence vehicle classes identified in Part 2
    # NOTE (verified 2026-09-03 against speechdnn/Noises actual tree, cross-checked
    # via panandicoding/Build-SE-Dataset, haoxiangsnr/UNetGAN-Demo, and
    # haoxiangsnr/Extremely-Low-SNR-Demo, all of which independently mirror the
    # same NoiseX-92 file set with identical names): four filenames previously
    # here did not match the real repo and would 404. Corrected below.
    FILES = [
        "leopard.wav",              # Leopard 1 tank engine / track noise
        "m109.wav",                 # M109 155mm self-propelled howitzer
        "f16.wav",                  # F-16 Falcon cockpit noise
        "destroyerengine.wav",      # Naval Destroyer engine room
        "destroyerops.wav",         # Naval Destroyer operations room
        "buccaneercockpit1.wav",    # Buccaneer jet cockpit (was: buccaneer1.wav — WRONG, fixed)
        "machinegun.wav",           # 0.50 caliber machine gun
        "factoryfloor1.wav",        # Heavy machinery / factory (was: factory1.wav — WRONG, fixed)
        "babble.wav",               # Background multi-speaker speech
        "pinknoise.wav",            # Pink noise reference (was: pink.wav — WRONG, fixed)
        "whitenoise.wav",           # White noise reference (was: white.wav — WRONG, fixed)
    ]

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results = []
        files_to_download = self.FILES[:4] if sample_mode else self.FILES

        for filename in files_to_download:
            url = f"{self.BASE_RAW_URL}/{filename}"
            dest = self.output_dir / filename
            res = self.download_file(url, dest, dry_run=dry_run)
            results.append(res)

        return results
