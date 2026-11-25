# Research results

[← ResearchBoost-Agent](../../README.md#results)

**27 projects · 7 research domains**

Each report includes the workload, hardware, timing and validation results, with links to the project’s code and experiments.

| Domain | Projects | Report |
|---|---|---|
| LLM | [NanoChat](https://github.com/AI4AI-for-Anything/Faster-NanoChat) by [@Bool1020](https://github.com/Bool1020), YOCO, [LLM2Vec](https://github.com/AI4AI-for-Anything/Faster-LLM2Vec) by [@Bool1020](https://github.com/Bool1020) | [Open report](llm.md) |
| MLLM | LLaVA, BAGEL, R3, [LLaVA-OneVision-2](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2) by [@GONGSHUKAI](https://github.com/GONGSHUKAI) | [Open report](multimodal.md) |
| Image Generation | [G-STAR](https://github.com/DAGroup-PKU/G-STAR), RAE, DMD2, JiT | [Open report](image-generation.md) |
| Video Generation | [Wan2.2-VSA](https://github.com/JoeyXuquant11/Faster-Wan22-VSA-MUSA) by [@JoeyXuquant11](https://github.com/JoeyXuquant11), [SANA-WM](https://github.com/JoeyXuquant11/Faster-SANA-WM-MUSA) by [@JoeyXuquant11](https://github.com/JoeyXuquant11), Self-Forcing, [MiniMax-H3](https://github.com/2018cx/Faster-MiniMax-H3) by [@2018cx](https://github.com/2018cx) | [Open report](video-generation.md) |
| Embodied AI | RoboTwin, Motus, FastWAM, [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM) by [@GONGSHUKAI](https://github.com/GONGSHUKAI), [SuperMimic](https://github.com/hwbhwbhwb/Faster-SuperMimic) by [@hwbhwbhwb](https://github.com/hwbhwbhwb) | [Open report](embodied-ai.md) |
| AI for Science | [ProteinMPNN](https://github.com/micky-li-hd/Faster-ProteinMPNN) by [@micky-li-hd](https://github.com/micky-li-hd), [Protenix](https://github.com/micky-li-hd/Faster-Protenix) by [@micky-li-hd](https://github.com/micky-li-hd), [FermiNet](https://github.com/sen-ye/faster-ferminet) by [@sen-ye](https://github.com/sen-ye), [State](https://github.com/YHDGD5/Faster-State) by [@YHDGD5](https://github.com/YHDGD5) | [Open report](ai4s.md) |
| 3D Reconstruction | [Original 3DGS](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost) by [@HaoqinHong](https://github.com/HaoqinHong), [PIDG](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost) by [@HaoqinHong](https://github.com/HaoqinHong), [VGGT](https://github.com/SuperFCR/Faster-VGGT) by [@SuperFCR](https://github.com/SuperFCR) | [Open report](3d-reconstruction.md) |

Speedups compare each workload with its own baseline. **est.** marks an extrapolated result.

Updated **2026-11-25** from the linked project reports and [website results](https://ai4ai-for-anything.github.io/#research), including G-STAR's inference **2.72–6.18× on one H200** (merged upstream in [G-STAR#2](https://github.com/DAGroup-PKU/G-STAR/pull/2)), R3’s **2.08× on 8 × H100**, MiniMax-H3’s repaired **1.8570× on 8 × H20**, OpenWAM’s **2.54–2.77× on 4 × H200** (merged upstream), LLaVA-OneVision-2’s **2.2–3.0× on Hopper and Blackwell** (merged upstream), Protenix’s manually validated **1.55× on 8 × H20**, FermiNet’s **1.25× on H20**, and Moore Threads MTT S5000 (MUSA) video generation cases **Wan2.2-VSA (5.28× on 8 × MTT S5000)** and **SANA-WM (4.85× on 4 × MTT S5000)**. LLM2Vec reports independent MNTP and SimCSE stage measurements; its validation limitations are documented in the [LLM report](llm.md#faster-llm2vec).

3D reconstruction cases: **1.13×** across complete Original 3DGS pipelines and **4.17×** for the PIDG module group on one A800, plus **Faster-VGGT** at **2.19×** (Tier N2, −31.7% peak VRAM) and **1.86×** (pure lossless Tier N1) on one A100; [measurement scope and judge statuses](3d-reconstruction.md).
