---
layout: gallery-item
title: "QKerasV3: Bringing Next-Gen Quantization-Aware Training to Keras 3 for
  multi-backend edge deployment"
summary: >-
  Deep learning models are getting smarter, but they are also getting heavier.
  If you’re working in the fastML ecosystem – trying to deploy cutting-edge
  neural networks onto edge devices, ASICs, or FPGAs – you know that
  floating-point math is a luxury you often can’t afford.


  You need Quantization-Aware Training (QAT). You need QKeras.


  But as the deep learning world evolved to Keras 3, standard QKeras was left waiting for an upgrade. That changes today. Welcome to QKerasV3: completely rewritten from the ground up to bring state-of-the-art quantization directly to Keras 3.
submitter: Marius Köppel
domain: QAT
image: https://koeppel123.de/assets/img/qkeras-logo-1400.webp
image_alt: QKerasV3 introduces drop-in quantized layers built specifically for
  the Keras 3 ecosystem
affiliation: ETH Zurich
tools_used:
  - hls4ml
  - QKeras
tags:
  - Quantization-aware training
review_status: draft
---
Deep learning models are getting smarter, but they are also getting *heavier*. If you’re working in the fastML ecosystem -- trying to deploy cutting-edge neural networks onto edge devices, ASICs, or FPGAs -- you know that floating-point math is a luxury you often can’t afford. 

You need Quantization-Aware Training (QAT). You need **QKeras**. 

But as the deep learning world evolved to Keras 3, standard QKeras was left waiting for an upgrade. That changes today. Welcome to **QKerasV3**: completely rewritten from the ground up to bring state-of-the-art quantization directly to Keras 3.

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/qkeras-logo.png" title="QKerasV3 Logo" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    QKerasV3 introduces drop-in quantized layers built specifically for the Keras 3 ecosystem.
</div>

### 🎛️ The Power of Multi-Backend
The biggest superpower of QKerasV3 is that it inherits the incredible multi-backend architecture of Keras 3. Whether your team is writing training loops in **TensorFlow**, running massive distributed jobs in **JAX**, or researching custom gradients in **PyTorch**, QKerasV3 works seamlessly across all of them.

---

## 🔬 The Case Study: LHC Jet Tagging at the Edge

To see the power of QKerasV3, let's explore a real-world physics challenge: the [HLS4ML LHC jet dataset](https://openml.org/search?type=data&id=42468) introduced by [Duarte et al. (2018)](https://arxiv.org/abs/1804.06913).

Jets are collimated sprays of particles produced during proton-proton collisions at the LHC. Identifying their origin in real time is a core task for LHC trigger systems, which must make decision inferences within a few microseconds. The goal is to classify each jet into one of five categories: Gluons (`g`), Light quarks (`q`), W bosons (`w`), Z bosons (`z`), or Top quarks (`t`).

Our dataset contains 16 high-level jet substructure observables. We will build two models:
1. A baseline **Full-Precision (32-bit floating-point) Keras 3 model**.
2. An optimized **6-bit Quantized model** using QKerasV3.

---

## 🛠️ Step 1: Data Preparation & The FP32 Baseline

First, let's pull our environment variables, grab the data from OpenML, preprocess it, and train a classical Keras 3 model.

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(0)
tf.random.set_seed(seed=0)

# Fetch the dataset
data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']

# One-hot encode targets & split data
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y = np.eye(5)[y_encoded]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale inputs
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

os.makedirs('../data/jet-tagging', exist_ok=True)
np.save('../data/jet-tagging/X_train_val.npy', X_train_val)
np.save('../data/jet-tagging/X_test.npy', X_test)
np.save('../data/jet-tagging/y_train_val.npy', y_train_val)
np.save('../data/jet-tagging/y_test.npy', y_test)
np.save('../data/jet-tagging/classes.npy', le.classes_)
```

Now we construct and train our baseline floating-point network using standard Keras 3 `Dense` layers:

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Standard FP32 Architecture
baseline_model = Sequential()
baseline_model.add(Dense(64, input_shape=(16,), name='fc1', activation='relu'))
baseline_model.add(Dense(32, name='fc2', activation='relu'))
baseline_model.add(Dense(32, name='fc3', activation='relu'))
baseline_model.add(Dense(5, name='output', activation='softmax'))

baseline_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
baseline_model.fit(X_train_val, y_train_val, batch_size=1024, epochs=20, validation_split=0.25, shuffle=True)

os.makedirs('../models', exist_ok=True)
baseline_model.save('../models/keras_model_part1.h5')
```

An accuracy of **~75%** is expected for this 5-class problem. Some classes (notably gluon vs. light quark) are physically very similar and genuinely hard to separate.

---

## 🛠️ Step 2: The QKerasV3 Quantized Model

If we compiled the baseline model directly to an FPGA using Post-Training Quantization (PTQ), its accuracy would degrade significantly below ~8 bits. To fix this, we apply **Quantization-Aware Training (QAT)** via QKerasV3.

We map the exact same architecture, but swap standard layers for `QDense` and `QActivation` constrained to a rigid 6-bit grid.

```python
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

# 6-bit Quantized Architecture
quantized_model = Sequential()
quantized_model.add(
    QDense(
        64,
        input_shape=(16,),
        name='fc1',
        kernel_quantizer=quantized_bits(6, 0, alpha=1),
        bias_quantizer=quantized_bits(6, 0, alpha=1),
    )
)
quantized_model.add(QActivation(activation=quantized_relu(6), name='relu1'))

quantized_model.add(
    QDense(
        32,
        name='fc2',
        kernel_quantizer=quantized_bits(6, 0, alpha=1),
        bias_quantizer=quantized_bits(6, 0, alpha=1),
    )
)
quantized_model.add(QActivation(activation=quantized_relu(6), name='relu2'))

quantized_model.add(
    QDense(
        32,
        name='fc3',
        kernel_quantizer=quantized_bits(6, 0, alpha=1),
        bias_quantizer=quantized_bits(6, 0, alpha=1),
    )
)
quantized_model.add(QActivation(activation=quantized_relu(6), name='relu3'))

quantized_model.add(
    QDense(
        5,
        name='output',
        kernel_quantizer=quantized_bits(6, 0, alpha=1),
        bias_quantizer=quantized_bits(6, 0, alpha=1),
    )
)
quantized_model.add(Activation(activation='softmax', name='softmax'))

quantized_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
quantized_model.fit(X_train_val, y_train_val, batch_size=1024, epochs=30, validation_split=0.25, shuffle=True)
quantized_model.save('../models/qkeras_model_part2.h5')
```

---

## 📉 Step 3: Side-by-Side Comparison & Hardware Conversion

Now, let's look at how they stack up. We convert our QKerasV3 model over to `hls4ml`, compile it, and compare the execution accuracies across the Baseline, the Quantized variant, and the final structural hardware emulation.

```python
import hls4ml
from sklearn.metrics import accuracy_score
from keras.models import load_model

# 1. Parse configuration and convert to hardware model
config = hls4ml.utils.config_from_keras_model(quantized_model, granularity='name', backend='Vitis')
config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'

hls_model = hls4ml.converters.convert_from_keras_model(
    quantized_model,
    hls_config=config,
    backend='Vitis',
    output_dir='../hls4ml_prjs/hls4ml_prj_qkeras_part2',
    part='xcu200-fsgd2104-2-e',
)
hls_model.compile()

# 2. Collect predictions
y_ref = load_model('../models/keras_model_part1.h5').predict(np.ascontiguousarray(X_test))
y_qkeras = quantized_model.predict(np.ascontiguousarray(X_test))
y_hls = hls_model.predict(np.ascontiguousarray(X_test))

# 3. Print side-by-side accuracies
print('Accuracy baseline:  {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_ref, axis=1))))
print('Accuracy quantized: {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_qkeras, axis=1))))
print('Accuracy hls4ml:    {}'.format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))
```

---

## 🚀 FPGA Synthesis and Resource Optimization

With validation complete, we run high-level synthesis (HLS) to generate our hardware description.

```python
# Run C-Synthesis
hls_model.build(csim=False)
hls4ml.report.read_vivado_report('../hls4ml_prjs/hls4ml_prj_qkeras_part2')
```

Comparing our hardware allocation here against unquantized models reveals a massive reduction in space. Because of our strict 6-bit constraint, every internal multiplication falls comfortably below the ~10-bit threshold. Consequently, Vivado can completely skip using expensive, limited physical DSP slices—mapping the arithmetic logic onto cheaper, highly parallel Look-Up Tables (LUTs) instead!

#### ⚠️ A Note on HLS Resource Estimation

The resource summaries provided by Vitis HLS directly following initial C-synthesis are strictly **estimates** driven by conservative internal models. They regularly overestimate LUT usage—sometimes by an order of magnitude.

For a true blueprint of your post-route performance, run a full **Vivado Logic Synthesis** (`vsynth`):

```python
# Execute full Vivado Logic Synthesis (Takes 10-20 minutes)
hls_model.build(reset=False, csim=False, vsynth=True)
hls4ml.report.read_vivado_report('../hls4ml_prjs/hls4ml_prj_qkeras_part2')
```

---

### 📖 Further Reading

For an in-depth dive into the underlying physics and engineering behind low-latency edge accelerators, check out these foundations:

* Duarte, Han, Harris et al., *"Fast inference of deep neural networks in FPGAs for particle physics"*, JINST 13 P07027 (2018), [arXiv:1804.06913](https://arxiv.org/abs/1804.06913)
* Coelho Jr., Kuusela, Zhuang et al., *"Ultra Low-latency, Low-area Inference Accelerators using Heterogeneous Deep Quantization with QKeras and hls4ml"*, arXiv (2020), [arXiv:2006.10159](http://arxiv.org/abs/2006.10159)

*Have you built something cool with QKerasV3? Tag us on GitHub or drop a PR in the fastML community!*
