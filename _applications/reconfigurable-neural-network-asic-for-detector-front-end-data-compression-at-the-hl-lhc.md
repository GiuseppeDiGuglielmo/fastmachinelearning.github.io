---
layout: gallery-item
title: Reconfigurable Neural Network ASIC for Detector Front-End Data Compression at the HL-LHC
summary: A radiation-tolerant neural network ASIC implementing a quantized autoencoder for low-latency lossy data compression in the CMS High-Granularity Calorimeter at the HL-LHC.
submitter: Giuseppe Di Guglielmo
domain: "LHC detector front-end"
image: /images/applications/hgcal_asic_econ.png
affiliation: Fermilab
tools_used:
  - hls4ml
  - QKeras
  - Catapult HLS
tags:
  - cms
  - hgcal
  - hl-lhc
  - asic
  - data compression
  - autoencoder
  - hls4ml
  - qkeras
  - edge-ai
  - radiation-tolerance
review_status: pending
---

The CMS High-Granularity Calorimeter (HGCAL), being built for the High-Luminosity LHC, is an imaging calorimeter with more than six million readout channels. Its high granularity provides unprecedented spatial information for particle reconstruction, but it also creates a major data-movement challenge: trigger data must be processed and transmitted at the 40 MHz LHC bunch-crossing rate. Because not all detector information can be sent off-detector at full granularity, part of the data reduction must happen directly in the front-end electronics.

This application demonstrates a reconfigurable neural network ASIC for front-end data compression in the CMS HGCAL trigger path. The target task is lossy compression of the energy pattern from a single HGCAL sensor module before transmission to the off-detector trigger electronics. Each silicon sensor module provides 48 trigger-cell charge values; the front-end concentrator ASIC must reduce this information while preserving the most important features of the detector energy profile.

<img src="{{ site.baseurl }}/images/applications/hgcal_asic_flow.png" width="500" alt="block diagram of the HGCAL trigger path showing where the on-detector encoder fits in the data flow" />

The machine-learning algorithm is based on an autoencoder.
In the intended front-end use case, only the encoder is implemented on-detector: it receives the normalized trigger-cell charge pattern and compresses the sensor image into a lower-dimensional representation.
The full autoencoder is used during training as a proxy objective, encouraging the latent representation to retain enough information to reconstruct the original energy pattern.
This makes the encoder a compact "shape encoder" for the local calorimeter energy distribution.

A central feature of the design is reconfigurability.
The ASIC architecture is fixed after fabrication, but the neural network weights are programmable through an on-chip I²C interface.
This allows different compression algorithms to be loaded for different detector regions, sensor geometries, occupancies, or changing detector and collider conditions.
In this way, the design preserves some of the flexibility usually associated with programmable logic while targeting the lower power and latency of a custom ASIC.

In terms of Fast ML technology, the neural network is trained with quantization-aware training using QKeras, which allows the bit-widths of weights and activations to be co-optimized with accuracy during training.
hls4ml is then used to translate the trained model into synthesizable C++ HLS code, targeting Siemens' Catapult HLS tool for ASIC-specific high-level synthesis.
For this work, hls4ml was extended beyond its original FPGA focus to support Catapult HLS and the LP CMOS 65 nm technology node.
A SystemVerilog RTL IP for a programmable I²C peripheral was integrated alongside the hls4ml-generated encoder block to enable on-chip weight reconfiguration.

<img src="{{ site.baseurl }}/images/applications/hgcal_asic_application.png" width="500" alt="autoencoder neural network architecture and data flow for the baseline encoder model" />

The digital implementation consists of three main blocks: a converter for input normalization, the hls4ml-generated encoder, and the I²C peripheral for weight configuration.
The real-time requirements are set by the LHC bunch-crossing period: the ASIC accepts a new input every 25 ns and produces a result within two bunch crossings, giving a total inference latency of 50 ns.
The design was implemented in a 65 nm low-power CMOS process and carried through synthesis and physical layout flows.

<img src="{{ site.baseurl }}/images/applications/hgcal_asic_floorplan.png" width="500" alt="design floor-plan with integrated converter, encoder and I²C peripheral occupying a total area of 3.6 mm²" />

Because the ASIC operates in the HL-LHC detector environment, radiation tolerance is a core part of the implementation strategy.
The design targets a total ionizing dose of approximately 200 Mrad.
Single-event effects are mitigated using triple modular redundancy: the encoder and converter datapath use triplicated registers with majority voting, while the I²C peripheral uses full module triplication with autocorrection to protect the stored neural network weights.

<img src="{{ site.baseurl }}/images/applications/hgcal_asic_tmr.png" width="500" alt="triple modular redundancy scheme: each register is triplicated with a majority voter for the encoder and converter datapath" />

The final simulated implementation achieves a latency of 50 ns, an energy consumption of 2.38 nJ per inference, a power consumption of 95 mW, and an area of 3.6 mm².
Compared with an estimated fully unrolled FPGA implementation, the ASIC provides more than an order-of-magnitude improvement in power while also reducing latency.
This is the first radiation-tolerant on-detector ASIC implementation of a neural network for particle physics, and demonstrates a complete Fast ML path from quantization-aware training through hls4ml code generation, Catapult HLS synthesis, RTL validation, physical design, and radiation-aware design techniques.
