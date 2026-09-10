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
    
    # Verified against speechdnn/Noises/master/NoiseX-92 repository (100% 200 OK):
    FILES = [
        "leopard.wav",              # Leopard 1 tank engine / track noise
        "m109.wav",                 # M109 155mm self-propelled howitzer
        "f16.wav",                  # F-16 Falcon cockpit noise
        "destroyerengine.wav",      # Naval Destroyer engine room
        "destroyerops.wav",         # Naval Destroyer operations room
        "buccaneer1.wav",           # Buccaneer jet cockpit
        "machinegun.wav",           # 0.50 caliber machine gun
        "factory1.wav",             # Heavy machinery / factory
        "babble.wav",               # Background multi-speaker speech
        "pink.wav",                 # Pink noise reference
        "white.wav",                # White noise reference
        "volvo.wav",                # Volvo car interior vehicle noise
    ]

    def fetch(self, sample_mode: bool = False, dry_run: bool = False) -> List[DownloadResult]:
        results = []
        files_to_download = self.FILES[:4] if sample_mode else self.FILES

        for filename in files_to_download:
            url = f"{self.BASE_RAW_URL}/{filename}"
            fallback_urls = [
                f"https://raw.githubusercontent.com/panandicoding/Build-SE-Dataset/master/NoiseX-92/{filename}",
                f"https://raw.githubusercontent.com/haoxiangsnr/UNetGAN-Demo/master/NoiseX-92/{filename}",
            ]
            dest = self.output_dir / filename
            res = self.download_file(url, dest, dry_run=dry_run, fallback_urls=fallback_urls)
            results.append(res)

        return results
