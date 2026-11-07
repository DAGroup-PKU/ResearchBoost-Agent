# Research results

[← ResearchBoost-Agent](../../README.md#results)

**20 projects · 7 research domains**

Each report includes the workload, hardware, timing and validation results, with links to the project’s code and experiments.

| Domain | Projects | Report |
|---|---|---|
| LLM | [NanoChat](https://github.com/AI4AI-for-Anything/Faster-NanoChat) by [@Bool1020](https://github.com/Bool1020), YOCO, [LLM2Vec](https://github.com/AI4AI-for-Anything/Faster-LLM2Vec) by [@Bool1020](https://github.com/Bool1020) | [Open report](llm.md) |
| MLLM | LLaVA, BAGEL, R3 | [Open report](multimodal.md) |
| Image Generation | RAE, DMD2, JiT | [Open report](image-generation.md) |
| Video Generation | Self-Forcing, [MiniMax-H3](https://github.com/2018cx/Faster-MiniMax-H3) by [@2018cx](https://github.com/2018cx) | [Open report](video-generation.md) |
| Embodied AI | RoboTwin, Motus, FastWAM, [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM) by [@GONGSHUKAI](https://github.com/GONGSHUKAI) | [Open report](embodied-ai.md) |
| AI for Science | [ProteinMPNN](https://github.com/micky-li-hd/Faster-ProteinMPNN) by [@micky-li-hd](https://github.com/micky-li-hd), [Protenix](https://github.com/micky-li-hd/Faster-Protenix) by [@micky-li-hd](https://github.com/micky-li-hd), [FermiNet](https://github.com/sen-ye/faster-ferminet) by [@sen-ye](https://github.com/sen-ye) | [Open report](ai4s.md) |
| 3D Reconstruction | [Original 3DGS](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost) by [@HaoqinHong](https://github.com/HaoqinHong), [PIDG](https://github.com/HaoqinHong/Physics-Informed-Deformable-Gaussian-Splatting-ResearchBoost) by [@HaoqinHong](https://github.com/HaoqinHong) | [Open report](3d-reconstruction.md) |

Speedups compare each workload with its own baseline. **est.** marks an extrapolated result.

Updated **2026-11-07** from the linked project reports and [website results](https://ai4ai-for-anything.github.io/#research), including R3’s **2.08× on 8 × H100**, MiniMax-H3’s repaired **1.8570× on 8 × H20**, OpenWAM’s **2.54–2.77× on 4 × H200** (merged upstream), Protenix’s manually validated **1.55× on 8 × H20**, and FermiNet’s **1.25× on H20**. LLM2Vec reports independent MNTP and SimCSE stage measurements; its validation limitations are documented in the [LLM report](llm.md#faster-llm2vec).

New 3D reconstruction cases: **1.13×** across complete Original 3DGS pipelines and **4.17×** for the PIDG module group on one A800; [measurement scope and explicit judge statuses](3d-reconstruction.md).
