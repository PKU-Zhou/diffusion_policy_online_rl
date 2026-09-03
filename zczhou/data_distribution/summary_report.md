# Training Data Distribution Analysis Report

## Overview

This report summarizes distribution statistics of weights (and gradients, if collected) during training.

## Weight Distribution Analysis

### policy Network

#### linear_1_b

- mean: 0.000234
- std: 0.013115
- min: -0.031386
- max: 0.035166
- median: -0.001923

![policy linear_1_b weight distribution](weight/policy/linear_1_b_hist.png)

#### linear_1_w

- mean: 0.001545
- std: 0.161528
- min: -0.449476
- max: 0.440757
- median: -0.006338

![policy linear_1_w weight distribution](weight/policy/linear_1_w_hist.png)

#### linear_b

- mean: 0.003595
- std: 0.012137
- min: -0.024479
- max: 0.034616
- median: 0.002509

![policy linear_b weight distribution](weight/policy/linear_b_hist.png)

#### linear_w

- mean: -0.000582
- std: 0.221228
- min: -0.560222
- max: 0.524496
- median: -0.005146

![policy linear_w weight distribution](weight/policy/linear_w_hist.png)

### q1 Network

#### linear_1_b

- mean: -0.004651
- std: 0.015451
- min: -0.114054
- max: 0.080009
- median: -0.002834

![q1 linear_1_b weight distribution](weight/q1/linear_1_b_hist.png)

#### linear_1_w

- mean: -0.001787
- std: 0.059350
- min: -0.382800
- max: 0.240448
- median: -0.001522

![q1 linear_1_w weight distribution](weight/q1/linear_1_w_hist.png)

#### linear_b

- mean: -0.002955
- std: 0.025789
- min: -0.146777
- max: 0.135912
- median: -0.001546

![q1 linear_b weight distribution](weight/q1/linear_b_hist.png)

#### linear_w

- mean: -0.004118
- std: 0.215822
- min: -1.000926
- max: 0.973694
- median: -0.003469

![q1 linear_w weight distribution](weight/q1/linear_w_hist.png)

### q2 Network

#### linear_1_b

- mean: -0.003686
- std: 0.015681
- min: -0.100814
- max: 0.068045
- median: -0.002025

![q2 linear_1_b weight distribution](weight/q2/linear_1_b_hist.png)

#### linear_1_w

- mean: -0.001601
- std: 0.059203
- min: -0.346359
- max: 0.287452
- median: -0.001413

![q2 linear_1_w weight distribution](weight/q2/linear_1_w_hist.png)

#### linear_b

- mean: -0.000201
- std: 0.026778
- min: -0.102973
- max: 0.198695
- median: 0.000952

![q2 linear_b weight distribution](weight/q2/linear_b_hist.png)

#### linear_w

- mean: -0.002180
- std: 0.217150
- min: -0.995984
- max: 1.007630
- median: -0.004290

![q2 linear_w weight distribution](weight/q2/linear_w_hist.png)

## Conclusions

1. Weight distribution characteristics
2. Training stability assessment
3. Cross-network differences
