# VLGDiff: A Vision-Language Guided Diffusion Model for Image Inpainting

**Shuling Zheng**, **Zhan Li***, **Zhanglu Chen**, **Yinglue Yang**, **Zhiheng Fu**, **Feng Zhang**

Department of Computer Science, Jinan University, Guangzhou, China

## Abstract

Text-guided image inpainting has achieved remarkable progress in recent years. However, it faces challenges in aligning with user-provided prompts while maintaining the visual rationality of the generated content. Existing methods often capture global scene context insufficiently and exhibit deviations in text-semantic understanding. To address these issues, we propose a **Vision-Language Guided Diffusion (VLGDiff)** model for image inpainting. To ensure the visual coherence of generated images, VLGDiff introduces a **Visual Prior Anchor (VPA)**, which guides the diffusion process by enhancing spatial information from known regions. In addition, a **Multimodal Prior Fusion Module (MPFM)** is designed to integrate visual and textual knowledge, in which an **Adaptive Spatial Cross Block (ASCB)** dynamically adjusts the balance between text guidance and visual cues according to the size of the inpainting region. Extensive experiments demonstrate that VLGDiff outperforms state-of-the-art methods in generating visually coherent and semantically consistent inpainting results.

---

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@InProceedings{10.1007/978-981-92-3513-1_32,
  author    = {Shuling Zheng and Zhan Li and Zhanglu Chen and Yinglue Yang and Zhiheng Fu and Feng Zhang},
  title     = {VLGDiff: A Vision-Language Guided Diffusion Model for Image Inpainting},
  booktitle = {Advanced Intelligent Computing Technology and Applications},
  publisher = {Springer Nature Singapore},
  address   = {Singapore},
  pages      = {383--394},
  year       = {2027},
  isbn       = {978-981-92-3513-1},
  doi        = {10.1007/978-981-92-3513-1_32}
}
```
