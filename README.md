# VLGDiff: A Vision-Language Guided Diffusion Model for Image Inpainting

**Shuling Zheng**, **Zhan Li***, **Zhanglu Chen**, **Yinglue Yang**, **Zhiheng Fu**, **Feng Zhang**

Department of Computer Science, Jinan University, Guangzhou, China

## Abstract

Text-guided image inpainting has achieved remarkable progress in recent years. However, it faces challenges in aligning with user-provided prompts while maintaining the visual rationality of the generated content. Existing methods often capture global scene context insufficiently and exhibit the de-viation in text-semantic understanding. To address these issues, we propose a Vision-Language Guided Diffusion (VLGDiff) model for image inpainting. To ensure the visual coherence of generated images, VLGDiff introduces a Visual Prior Anchor (VPA), which guides the diffusion process by enhancing spatial information from known regions. In addition, a Multimodal Prior Fu-sion Module (MPFM) is designed to integrate visual and textual knowledge, in which an Adaptive Spatial Cross Block (ASCB) adjusts ratios of the text score and visual cues according to inpainting region sizes. Extensive experi-ments demonstrate that our VLGDiff outperforms state-of-the-art methods in generating visually rational and semantically consistent results.

---

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@InProceedings{10.1007/978-981-92-3513-1_32,
  author    = {Zheng, Shuling
               and Li, Zhan
               and Chen, Zhanglu
               and Yang, Yinglue
               and Fu, Zhiheng
               and Zhang, Feng},
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
