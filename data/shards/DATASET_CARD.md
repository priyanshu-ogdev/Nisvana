---
pretty_name: Project AEGIS Defence ANC Training Corpus
license: other  # mixed per-source, see License Ledger below — NOT uniformly redistributable
task_categories:
  - audio-to-audio
  - audio-classification
language:
  - en
tags:
  - speech-enhancement
  - noise-cancellation
  - defence-audio
  - webdataset
---

# Project AEGIS — Defence ANC Training Corpus

Auto-generated dataset card. Regenerate with `python -m data_forge export --card` after any change to `bibliography.py` or `config.py` — do not hand-edit this file.

## Audio Standard

- Sample rate: 48000 Hz (fullband)
- Loudness: -23.0 LUFS
- Format: mono WAV, 16-bit PCM

## Sources and Sync Tiers

| Source | Native rate | Sync tier | Augmentation policy | License |
|---|---|---|---|---|
| NOISEX-92 | 19980 Hz | TIER_3_UPSAMPLED_16K | moderate_no_pitch | Public Research / NATO RSG.10 |
| SHAReD | 48000 Hz | TIER_1_NATIVE_48K | moderate_blast_window | CC BY 4.0 |
| Gunshot Triangulation Dataset | 44100 Hz | TIER_2_RESAMPLED_44K | moderate_no_pitch | CC0 1.0 Universal |
| DroneAudioSet | 48000 Hz | TIER_1_NATIVE_48K | minimal_level_crop | MIT |
| Military Audio Dataset (MAD) | 48000 Hz | TIER_1_NATIVE_48K | minimal_level_crop | CC BY-SA 4.0 |
| VoiceBank-DEMAND / CSTR VCTK | 48000 Hz | TIER_1_NATIVE_48K | none | CC BY 4.0 |
| DNS Challenge (Interspeech / ICASSP) | 48000 Hz | TIER_1_NATIVE_48K | minimal_snr_only | CC BY 4.0 / Microsoft Research |
| AEC Challenge (ICASSP) | 48000 Hz | TIER_1_NATIVE_48K | none | CC BY 4.0 / Microsoft Research |
| UrbanSound8K / ESC-50 Sirens | 44100 Hz | TIER_2_RESAMPLED_44K | moderate_full | CC BY-NC 3.0 |
| OpenSLR-28 Room Impulse Responses | 16000 Hz | TIER_3_UPSAMPLED_16K | none | Apache 2.0 |
| MUSAN | 16000 Hz | TIER_3_UPSAMPLED_16K | none | Creative Commons Attribution 4.0 |

## Full Citation Ledger

- **The INTERSPEECH 2020 Deep Noise Suppression Challenge** — Chandan K. A. Reddy et al.. *INTERSPEECH 2020*. https://github.com/microsoft/DNS-Challenge (CC BY 4.0 / Microsoft Research). Role: High-volume 48kHz native clean speech targets and noise corpus
- **CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit** — Christophe Veaux, Junichi Yamagishi, Kirsten MacDonald. *University of Edinburgh, 2017*. https://datashare.ed.ac.uk/handle/10283/2950 (CC BY 4.0). Role: High-quality 48kHz native multi-speaker clean speech ground truth
- **DEMAND: Diverse Environments Multichannel Acoustic Noise Database** — Joachim Thiemann, Nobutaka Ito, Emmanuel Vincent. *Proceedings of Meetings on Acoustics (POMA), 2013*. https://zenodo.org/record/1227121 (CC BY 4.0). Role: 16-channel and multichannel environmental noise benchmark
- **FSD50K: An Open Dataset of Human-Labeled Sound Events** — Eduardo Fonseca et al.. *arXiv:2010.00475, 2020*. https://zenodo.org/record/4060432 (CC BY 4.0 / CC0). Role: Open sound event baseline for varied background acoustic conditions
- **AudioSet: An Ontology and Human-Labeled Dataset for Audio Events** — Jort F. Gemmeke et al.. *ICASSP 2017*. https://research.google.com/audioset/ (CC BY 4.0). Role: Ontology crosswalk and large-scale noise variety
- **MUSAN: A Music, Speech, and Noise Corpus** — David Snyder, Guoguo Chen, Daniel Povey. *arXiv:1510.08484 / OpenSLR 17, 2015*. https://www.openslr.org/17/ (CC BY 4.0). Role: Strictly reserved for generalization testing (unseen noise evaluation)
- **A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition** — Tom Ko, Vijayaditya Peddinti, Daniel Povey, Michael L. Seltzer, Sanjeev Khudanpur. *ICASSP 2017*. https://www.openslr.org/28/ (Apache 2.0). Role: Room Impulse Responses (RIRs) for dynamic physical acoustic convolution
- **A Military Audio Dataset for Situational Awareness and Surveillance** — June-Woo Kim, Ji-Hwan Yoon, Sang-Min Jung. *Scientific Data 11:668 (Nature), 2024*. https://github.com/kaen2891/military_audio_dataset (CC BY-SA 4.0). Role: Real operational military audio (tracked vehicles, shelling, combat)
- **Assessment for Automatic Speech Recognition: II. NOISEX-92: A Database and an Experiment to Study the Effect of Additive Noise on Speech Recognition Systems** — Andrew P. Varga, Herman J. M. Steeneken. *Speech Communication 12(3):247-251, NATO RSG.10, 1993*. https://spib.linse.ufsc.br/noisex.html (Public Research / NATO RSG.10). Role: Irreplaceable defence vehicle recordings: Leopard tank, M109 howitzer, F-16 cockpit, Destroyer engine
- **Data: Gunshots recorded in an open field using iPod Touch devices (Gunshot Triangulation Dataset)** — William Cooper, Ryan Shaw. *Dryad, 2020*. https://doi.org/10.5061/dryad.wm37pvmkc (CC0 1.0 Universal). Role: Multi-firearm ballistics recordings with known muzzle-to-mic geometry
- **Gunshot Audio Forensics Dataset** — Cadre Research Labs. *NIJ Grant 2016-DN-BX-0183, 2016*. https://cadreforensics.com/audio/ (Academic Research). Role: High-speed ballistic acoustic muzzle blast and supersonic shockwave ground truth
- **A Multi-Firearm, Multi-Orientation Audio Dataset of Gunshots** — Donald Kabealo et al.. *Data in Brief 48:109091, 2023*. https://doi.org/10.1016/j.dib.2023.109091 (CC BY 4.0). Role: Directional gunshot acoustic profile across diverse firearm calibers
- **Explosion Detection Using Smartphones: Ensemble Learning with the Smartphone High-Explosive Audio Recordings Dataset (SHAReD)** — S. K. Takazawa, S. K. Popenhagen, L. A. Ocampo Giraldo, J. D. Hix, S. J. Thompson, D. L. Chichester, C. P. Zeiler, M. A. Garcés. *Sensors 24(20):6688, 2024 (Dataverse doi:10.7910/DVN/ROWODP)*. https://doi.org/10.7910/DVN/ROWODP (CC BY 4.0). Role: 326 real high-explosive blast and detonation waveforms
- **DroneAudioSet: An Audio Dataset for Drone Detection and Acoustic Analysis** — ahlab-drone-project. *arXiv:2308.10659 / Hugging Face, 2023*. https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet (MIT). Role: 23.5h of annotated drone ego-noise and multi-rotor acoustic signatures
- **A Dataset and Taxonomy for Urban Sound Research (UrbanSound8K)** — Justin Salamon, Christopher Jacoby, Juan Pablo Bello. *ACM Multimedia 2014*. https://urbansounddataset.feedback.acm.org/ (CC BY-NC 3.0). Role: Emergency vehicle siren subset for urban hazard awareness
- **ESC: Dataset for Environmental Sound Classification** — Karol J. Piczak. *ACM Multimedia 2015*. https://github.com/karolpiczak/ESC-50 (CC BY-NC 3.0). Role: Siren and wind/rotor benchmark recordings
- **Acoustic Echo Cancellation Challenge (ICASSP)** — Ross Cutler et al., Microsoft Research. *ICASSP 2021, 2022, 2023*. https://github.com/microsoft/AEC-Challenge (CC BY 4.0 / Microsoft Research). Role: Dedicated paired near-end, far-end, and acoustic echo for Model 5 (Gated AEC)
- **DeepFilterNet: Low Complexity Speech Enhancement Framework for Full-Band Audio** — Hendrik Schröter, Tobias Rosenkranz, Alberto N. Escalante-B., Pascal Maier. *IEEE/ACM TASLP 2022 & ICASSP 2023*. https://github.com/Rikorose/DeepFilterNet (MIT / Apache 2.0). Role: Target Model 1 & 2 fullband speech enhancement
- **CleanUMamba: Speech Enhancement with State Space Models** — R. Groot, Y. Chen, J. van Gemert, W. Gao. *arXiv:2410.11062, 2024*. https://arxiv.org/abs/2410.11062 (Research Open Access). Role: Target Model 3 state-space speech enhancement
- **Data Augmentation for Pathological Speech Enhancement: A Comparative Study of Noise, Transformative, and Generative Methods** — Anonymous / Speech Enhancement Consortium. *arXiv:2602.14671, 2026*. https://arxiv.org/abs/2602.14671 (Open Access). Role: Grounding principle: Noise augmentation dominates; pitch-shifting degrades; generative data harms
- **Formant Disruption and Pitch-Shift Invariance in Neural Audio Representation** — Takuhiro Kaneko et al.. *arXiv:2407.05471, 2024*. https://arxiv.org/abs/2407.05471 (Open Access). Role: Demonstration that pitch-shifting speech corrupts vocal tract formants
- **UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022** — Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, Hiroshi Saruwatari. *arXiv:2204.02152, 2022*. https://arxiv.org/abs/2204.02152 (Open Access). Role: Perceptually-bounded augmentation range principles

## Known Limitations (disclosed, not synthesized around)

- Rotor/helicopter and wind classes remain thin on real recordings; no synthetic/procedural data is used anywhere in this corpus by design.
- NOISEX-92-derived classes (armored vehicle, naval, jet) are content-authentic but duration-limited (minutes per platform, not hours).
- CC BY-NC sources (ESC-50, portions of UrbanSound8K/FSD50K) are excluded under `--commercial-strict` preprocessing mode; included only for research/prototype use otherwise.
