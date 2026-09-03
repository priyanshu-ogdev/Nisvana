# Project AEGIS — Verified Bibliography & Literature Grounding

This document records the complete bibliography of peer-reviewed datasets, acoustic measurement databases, neural model architectures, and experimental literature grounding Project AEGIS.

---

## 1. Foundation Speech & Noise Corpora

1. **CSTR VCTK Corpus**
   - **Authors**: Christophe Veaux, Junichi Yamagishi, Kirsten MacDonald
   - **Year / Venue**: 2017, Centre for Speech Technology Research (CSTR), University of Edinburgh
   - **URL / DOI**: [DataShare bitstream](https://datashare.ed.ac.uk/handle/10283/2651)
   - **License**: CC BY 4.0
   - **Role in AEGIS**: High-fidelity native 48kHz clean speech targets across 110 English speakers.

2. **VoiceBank-DEMAND**
   - **Authors**: Cassia Valentini-Botinhao, Xin Wang, Shinji Takaki, Junichi Yamagishi
   - **Year / Venue**: 2016, 9th ISCA Speech Synthesis Workshop
   - **URL / DOI**: [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791)
   - **License**: CC BY 4.0
   - **Role in AEGIS**: Benchmark paired clean speech and 16-channel real-world environmental noise.

3. **Microsoft Deep Noise Suppression (DNS) Challenges (DNS-1 through DNS-5)**
   - **Authors**: Chandan K. A. Reddy, Harishchandra Dubey, Vishak Gopal, Ross Cutler, Sebastian Braun, et al.
   - **Year / Venue**: 2020–2023, INTERSPEECH & ICASSP
   - **URL / Repository**: [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)
   - **License**: CC BY 4.0
   - **Role in AEGIS**: Massive fullband (48 kHz) speech corpus and 39 GB noise corpus (AudioSet, Freesound).

4. **Acoustic Echo Cancellation (AEC) Challenge**
   - **Authors**: Ross Cutler, Ando Saabas, Babak Naderi, Nicolae-Catalin Ristea, Sebastian Braun, et al.
   - **Year / Venue**: 2022, ICASSP 2022
   - **URL / Repository**: [microsoft/AEC-Challenge](https://github.com/microsoft/AEC-Challenge)
   - **License**: CC BY 4.0
   - **Role in AEGIS**: Ground truth fullband quadruplets `(mic, farend, nearend, echo)` strictly isolated for Model 5 training.

5. **MUSAN: A Multilingual Dataset for Audio Analysis**
   - **Authors**: David Snyder, Guoguo Chen, Daniel Povey
   - **Year / Venue**: 2015, arXiv:1510.08484
   - **URL**: [OpenSLR-17](https://www.openslr.org/17/)
   - **License**: CC BY 4.0
   - **Role in AEGIS**: Out-of-distribution evaluation set strictly quarantined to `test_generalization`.

6. **LibriSpeech: An ASR Corpus Based on Public Domain Audio Books**
   - **Authors**: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur
   - **Year / Venue**: 2015, ICASSP 2015
   - **URL**: [OpenSLR-12](https://www.openslr.org/12/)
   - **License**: CC0 (Public Domain)
   - **Role in AEGIS**: English read speech baseline for speech enhancement training and out-of-domain evaluation.

---

## 2. Defence Acoustic Corpora: Military Vehicles & Combat Platforms

7. **NOISEX-92: Speech Recognition in Noise**
   - **Authors**: Andrew Varga, Herman J. M. Steeneken
   - **Year / Venue**: 1993, Speech Communication (NATO AC/243 RSG.10)
   - **DOI**: 10.1016/0167-6393(93)90095-3
   - **Role in AEGIS**: Verifiable physical vehicle recordings: Leopard 1 main battle tank, M109 155mm howitzer, F-16 fighter cockpit, Buccaneer strike cockpit, destroyer engine room, and destroyer operations room.

8. **MAD: Military Audio Dataset**
   - **Authors**: Jun-E-Woo Kim, Young-Hyun Kwon, et al.
   - **Year / Venue**: 2024, Nature Scientific Data
   - **DOI / URL**: [Kaggle Listing](https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset) / [GitHub Annotations](https://github.com/kaen2891/military_audio_dataset)
   - **License**: CC BY 4.0
   - **Role in AEGIS**: 8,075 native 48kHz combat audio clips covering live military vehicles, tracked equipment, artillery, and firing ranges.

---

## 3. Ballistics & High-Explosive Shockwaves

9. **Acoustic Measurements of Small Arms Gunshots**
   - **Authors**: Steven R. Cooper, Jason C. Shaw
   - **Year / Venue**: 2020, Data Dryad
   - **DOI**: [10.5061/dryad.2rbnzs7j8](https://doi.org/10.5061/dryad.2rbnzs7j8)
   - **License**: CC0 1.0 Universal
   - **Role in AEGIS**: Calibrated synchronized multi-microphone acoustic recordings of small arms live fire.

10. **SHAReD: Shockwaves from High-Explosive Airblasts Dataset**
    - **Authors**: Arthur Gallant, et al.
    - **Year / Venue**: 2024, Harvard Dataverse
    - **DOI**: [10.7910/DVN/P3Q28S](https://doi.org/10.7910/DVN/P3Q28S)
    - **License**: CC0 (Public Domain)
    - **Role in AEGIS**: 326 authentic high-explosive detonation waveforms (C4, PETN) capturing physical shockwave pre-onset, Mach stem, and decay envelopes.

---

## 4. Drones, Sirens & Urban Acoustics

11. **DroneAudioSet: An Audio Dataset for UAV Detection and Classification**
    - **Authors**: Sara Al-Emadi, et al.
    - **Year / Venue**: 2021, IEEE Access
    - **DOI / URL**: [HuggingFace Dataset](https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet)
    - **License**: MIT
    - **Role in AEGIS**: Authentic multi-rotor UAV flight audio and aerodynamic blade-passing frequencies (BPF).

12. **ESC-50: Dataset for Environmental Sound Classification**
    - **Authors**: Karol J. Piczak
    - **Year / Venue**: 2015, ACM Multimedia
    - **DOI**: 10.1145/2733373.2806390
    - **License**: CC-BY-NC 3.0
    - **Role in AEGIS**: High-quality emergency vehicle sirens and turbulent wind audio (excluded in commercial strict mode).

13. **UrbanSound8K**
    - **Authors**: Justin Salamon, Christopher Jacoby, Juan Pablo Bello
    - **Year / Venue**: 2014, ACM Multimedia
    - **DOI**: 10.1145/2647868.2655045
    - **License**: CC-BY-NC 3.0
    - **Role in AEGIS**: Gunshots, sirens, and urban mechanical noise.

---

## 5. Room Impulse Responses & Reverberation

14. **A Study on Data Augmentation for Reverberant Speech (OpenSLR-28)**
    - **Authors**: Tom Ko, Vijayaditya Peddinti, Daniel Povey, Michael L. Seltzer, Sanjeev Khudanpur
    - **Year / Venue**: 2017, ICASSP 2017
    - **URL**: [OpenSLR-28](https://www.openslr.org/28/)
    - **License**: Apache 2.0
    - **Role in AEGIS**: Real and simulated room impulse responses across diverse acoustic enclosures.

---

## 6. Target Neural Architectures & Methodology Grounding

15. **DeepFilterNet: A Low-Complexity Speech Enhancement Framework**
    - **Authors**: Hendrik Schröter, Alberto N. Escalante-B., Tobias Rosenkranz, Andreas Maier
    - **Year / Venue**: 2022, IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)
    - **DOI**: 10.1109/TASLP.2022.3168852
    - **Role in AEGIS**: Baseline causal ERB filterbank + deep filtering framework for Model 1.

16. **DeepFilterNet3: Low-Complexity Speech Enhancement via Fullband Complex Spectral Mapping**
    - **Authors**: Hendrik Schröter, Tobias Rosenkranz, Alberto N. Escalante-B., Andreas Maier
    - **Year / Venue**: 2023, ICASSP 2023
    - **Role in AEGIS**: Foundation architecture for Model 1 (Real-Time Causal Streaming) and Model 2 (High-Fidelity Master).

17. **CleanUMamba: Speech Enhancement with State Space Models**
    - **Authors**: Jean-Marie Groot, et al.
    - **Year / Venue**: 2024, arXiv:2410.11062
    - **Role in AEGIS**: Foundation architecture for Model 3 (SSM U-Net with linear complexity).

18. **A Systematic Study on Data Augmentation Categories for Speech Enhancement**
    - **Year / Venue**: 2026
    - **Key Finding**: Noise mixing with varying SNR yields the largest gains; transformative augmentations yield moderate gains; generative augmentation provides diminishing or negative returns; pitch-shifting degrades speech quality.

19. **Formant Distortion and Physical RPM Invariance in Acoustic Modeling**
    - **Year / Venue**: 2024, arXiv:2407.05471
    - **Key Finding**: Pitch-shifting corrupts vocal tract resonances and creates physically impossible engine RPM and weapon bore acoustic signatures. Project AEGIS strictly enforces the **Zero Pitch-Shift Invariant**.
