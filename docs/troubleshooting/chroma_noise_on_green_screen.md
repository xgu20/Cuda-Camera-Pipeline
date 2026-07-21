# Troubleshooting chroma noise on a green screen

Large, nearly uniform colored regions can expose mottled red/blue chroma noise.
The artifact usually comes from sensor noise being amplified and spatially
spread by several ISP stages rather than from one isolated block.

## Reproduce

If the optional local Infinite capture is available:

```bash
./build/libreisp \
  data/infinite/Indoor1_2592x1536_10bit_GRBG.raw \
  data/infinite/Indoor1_2592x1536_10bit_GRBG.png
```

Inspect the large green felt region near the upper-left of the result. This
capture is not distributed by the repository; see [data acquisition](../../data/README.md).

## Isolate the source

Change one stage at a time:

1. Replace the CCM with an identity matrix. A large reduction indicates that
   saturation gain in the CCM is amplifying chroma noise.
2. Set all R/Gr/Gb/B white-balance gains to 1.0. A reduction in red and blue
   speckle indicates amplification of low-SNR R/B channels.
3. Compare dumps before and after demosaic. Pixel-scale Bayer noise becoming
   broad color patches suggests interpolation is spreading high-frequency
   noise spatially.
4. Bypass LSC. If edge noise falls to the level seen at the image center, high
   lens-shading gains are the main edge amplifier.
5. Bypass YUV denoise and edge enhancement independently. This separates the
   denoiser's suppression from sharpening-related noise amplification.

## Typical causes

| Stage | Mechanism | Effect |
| --- | --- | --- |
| Sensor | Shot, read, and thermal noise | Initial physical noise source |
| White balance | High gain on weak R/B channels under green illumination | Amplified R/B noise |
| Demosaic | Neighborhood interpolation spreads random Bayer samples | Larger, more visible color patches |
| CCM | High diagonal and negative off-diagonal coefficients increase saturation | Strong chroma-noise amplification |
| LSC | Edge gains compensate lens falloff | More noise near image edges |
| Edge enhancement | High-frequency boost cannot distinguish all noise from detail | More visible residual noise |

## Mitigations

- Add CFA-aware Bayer noise reduction before demosaic to prevent spatial spread
  at the source.
- Tune the existing YUV denoiser with stronger U/V filtering than luma
  filtering. Human vision generally tolerates more chroma smoothing.
- Use ISO- or gain-dependent CCM and sharpening strength.
- Apply chroma suppression in dark or highly saturated regions.
- Improve AWB statistics with saturation rejection, gain limits, confidence,
  and temporal smoothing.

The current pipeline already contains RGB-to-YUV conversion, YUV denoise, and
edge enhancement. Their parameters come from the sensor sidecar or
`config/golden_tuning.json`; see [include/blocks.h](../../include/blocks.h) and
[the denoise algorithm notes](../algorithms/yuv_denoise.md).
