---
layout: gallery-item
title: Reconfigurable Neural Network ASIC for Detector Front-End Data Compression at the HL-LHC
summary: A radiation-tolerant neural network ASIC implementing a quantized autoencoder for low-latency lossy data compression in the CMS High-Granularity Calorimeter at the HL-LHC.
submitter: Giuseppe Di Guglielmo
domain: "LHC detector front-end"
image: /images/applications/hgcal_asic_flow.png
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

The CMS High-Granularity Calorimeter (HGCAL), being built for the High-Luminosity LHC, is an imaging calorimeter with more than six million readout channels.
Its high granularity provides unprecedented spatial information for particle reconstruction, but it also creates a major data-movement challenge: trigger data must be processed and transmitted at the 40 MHz LHC bunch-crossing rate.
Because not all detector information can be sent off-detector at full granularity, part of the data reduction must happen directly in the front-end electronics.

This application demonstrates a reconfigurable neural network ASIC for front-end data compression in the CMS HGCAL trigger path.
The target task is lossy compression of the energy pattern from a single HGCAL sensor module before transmission to the off-detector trigger electronics.
Each silicon sensor module provides 48 trigger-cell charge values; the front-end concentrator ASIC must reduce this information while preserving the most important features of the detector energy profile.

<img src="/images/applications/hgcal_asic_flow.png" width="500" alt="block diagram of the HGCAL trigger path showing where the on-detector encoder fits in the data flow" />

The machine-learning algorithm is based on an autoencoder.
In the intended front-end use case, only the encoder is implemented on-detector: it receives the normalized trigger-cell charge pattern and compresses the sensor image into a lower-dimensional representation.
The full autoencoder is used during training as a proxy objective, encouraging the latent representation to retain enough information to reconstruct the original energy pattern.
This makes the encoder a compact "shape encoder" for the local calorimeter energy distribution.

A central feature of the design is reconfigurability.
The ASIC architecture is fixed after fabrication, but the neural network weights are programmable through an on-chip I²C interface.
This allows different compression algorithms to be loaded for different detector regions, sensor geometries, occupancies, or changing detector and collider conditions.
In this way, the design preserves some of the flexibility usually associated with programmable logic while targeting the lower power and latency of a custom ASIC.

<img src="/images/applications/PLACEHOLDER_hgcal_asic_architecture.png" width="500" alt="[PLACEHOLDER: ASIC architecture diagram showing the three main digital blocks: input converter, hls4ml encoder, and I2C configuration peripheral]" />

The neural network was trained with quantization-aware training using QKeras, then translated to hardware with hls4ml and Catapult HLS.
The baseline encoder combines a convolutional layer and a dense layer, using fixed-point arithmetic to meet the tight area, power, and latency constraints of front-end detector electronics.
The digital implementation includes three main blocks: a converter for input normalization, the hls4ml-generated encoder, and an I²C peripheral used to configure the neural network weights.

The real-time requirements are set by the LHC bunch-crossing period.
The ASIC accepts a new input every 25 ns and adds a total inference latency of two bunch crossings, corresponding to 50 ns.
The design was implemented in a 65 nm low-power CMOS process and processed through synthesis and physical layout flows.

<img src="/images/applications/PLACEHOLDER_hgcal_asic_floorplan.png" width="500" alt="[PLACEHOLDER: physical layout floorplan of the ASIC showing the placement of the three digital blocks in 65 nm CMOS]" />

Because the ASIC is intended for the HL-LHC detector environment, radiation tolerance is part of the implementation strategy.
The design targets a total ionizing dose environment of approximately 200 Mrad.
Single-event effects are mitigated using triple modular redundancy: the encoder and converter datapath use triplicated registers with majority voting, while the I²C peripheral, which stores the neural-network parameters, uses full module triplication with autocorrection.

<img src="/images/applications/PLACEHOLDER_hgcal_asic_tmr.png" width="500" alt="[PLACEHOLDER: diagram illustrating the triple modular redundancy strategy applied to the datapath registers and the I2C peripheral]" />

The final simulated implementation achieves a latency of 50 ns, an energy consumption of 2.38 nJ per inference, a power consumption of 95 mW, and an area of 3.6 mm².
Compared with an estimated fully unrolled FPGA implementation, the ASIC provides more than an order-of-magnitude improvement in power while also reducing latency.
This result demonstrates the potential of embedding machine learning directly in detector front-end ASICs, where data compression, low latency, low power, and radiation tolerance must be addressed together.

This work is the first radiation-tolerant on-detector ASIC implementation of a neural network for particle physics, and represents a milestone for Fast ML moving from off-detector FPGA trigger systems into on-detector custom silicon.
It shows a complete path from quantized neural network training to ASIC implementation, including hls4ml-based code generation, HLS, RTL validation, physical design, power analysis, and radiation-aware design techniques.
