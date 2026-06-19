---
layout: gallery-item
title: Anomaly Detection for New Physics at CMS Level 1 Trigger
summary: AXOL1TL (Anomaly eXtraction Online Level-1 Trigger Lightweight) is an
  ultra low latency anomaly detection algorithm deployed at the CMS experiment
  Level-1 Trigger.
submitter: Sioni Summers
domain: "LHC trigger "
image: https://twiki.cern.ch/twiki/pub/CMSPublic/AXOL1TL2023/event_display.png
affiliation: CERN
tools_used:
  - hls4ml
  - HGQ
  - da4ml
  - QKeras
tags:
  - cms
  - " cern"
  - anomaly detection
  - trigger
  - "hls4ml "
review_status: approved
---

AXOL1TL (Anomaly eXtraction Online Level-1 Trigger Lightweight) is an ultra low latency anomaly detection algorithm deployed at the CMS experiment Level-1 Trigger.
The CMS Level-1 Trigger must process each LHC collision event - at a rate of 40 MHz collisions - and decide whether to keep or reject them for further processing before they are available to physicists to search for new physics.
This trigger system comprises many Xilinx Virtex 7 FPGAs for processing and decision making.
AXOL1TL targets events that look anomalous compared to the bulk of background data, making sure that they are selected for readout.
The algorithm resides in the final decision making boards of the system, where "high level" physics information has already been reconstructed - particle properties and composite objects like jets and missing transverse energy.

<img src="https://www.hep.ph.ic.ac.uk/mp7/images/pages/IMG_2170_COMP_ADJ.jpg" width="500" />

The first NN model for AXOL1TL was a Variational AutoEncoder (VAE) trained on real background data from CMS - so called "Zero Bias" data.
This is data that is triggered without applying a physics-based trigger filter, making it ideal for training an algorithm designed to be unbiased towards new physics.
In order to fit into the existing hardware FPGA system, the NN inference must take place with a latency of only 50 ns, so the model needed to be extremely shallow. 

<img src="https://twiki.cern.ch/twiki/pub/CMSPublic/AXOL1TL2025/axol1tl_v4_diagram.png" width="500" />

The conventional way of using a VAE for anomaly detection is to compress the inputs to a bottleneck, then reconstruct them back to the original inputs, i.e. to learn `f(x) = x` where `f` is a Neural Network.
The premise is that the quality of that reconstruction will perform well for frequently seen data, but poorly for rare data - the anomalies.
By defining an anomaly score `s = ||x - f(x)||²` we can compute a single number to use for the trigger decision.
In order to fit within the tight latency budget, we cropped the reconstruction part of the VAE and instead perform anomaly detection in the latent space - the `μ²` term in the diagram above.
The second variant of the model first of all trained an embedding using a contrastive training to a low-dimensional latent space, then trained a VAE to compress and reconstruct that low-dimensional latent space.

<img src="https://twiki.cern.ch/twiki/pub/CMSPublic/AXOL1TL2025/axol1tl_v5_diagram.png" width="500" />

In terms of Fast ML technology, the algorithm has used multiple FastML tools over different model versions.
The first deployment in 2024 used QKeras for training, and hls4ml Vitis backend, and Vitis HLS for FPGA code generation. 
The later, deeper, model was trained using HGQ, and deployed with da4ml and Vitis HLS. 
A custom top-level HLS wrapper was written to interface the hls4ml NN module to the VHDL framework of the system hosting the algorithm.
The wrapper maps the `struct` data provided by the host system to the array of uniform precision data expected by hls4ml.
It also enforces full inlining of all code, to help reach the tight 50 ns at 40 MHz requirement.

<img src="https://twiki.cern.ch/twiki/pub/CMSPublic/AXOL1TL2025/axol1tl_floorplan.png" width="200" />

AXOL1TL has taken data at CMS throughout 2024, 2025, and 2026 data taking periods.
We predict how efficient AXOL1TL is at detecting some hypothesised new physics scenarios by evaluating it on simulated examples of those scenarios.
The amount of trigger rate that is allocated to AXOL1TL is then allocated by the CMS collaboration.

<img src="https://twiki.cern.ch/twiki/pub/CMSPublic/AXOL1TL2025/axol1tl_roc_v4v5.png" width="500" />

The physics analysis of the collected data is ongoing!
Here's one of the events we selected, which was the event with the highest anomaly score that wouldn't have been picked up by another trigger decision.

<img src="https://twiki.cern.ch/twiki/pub/CMSPublic/AXOL1TL2023/event_display.png" width="500" />
