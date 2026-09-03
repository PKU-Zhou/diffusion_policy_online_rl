# Training Data Distribution Analysis Report

## Overview

This report summarizes distribution statistics of weights (and gradients, if collected) during training.

## Weight Distribution Analysis

### policy Network

#### linear_1_b

- mean: 0.000714
- std: 0.005433
- min: -0.008349
- max: 0.012455
- median: 0.000682

![policy linear_1_b weight distribution](weight/policy/linear_1_b_hist.png)

#### linear_1_w

- mean: 0.001834
- std: 0.159693
- min: -0.358241
- max: 0.360036
- median: -0.003397

![policy linear_1_w weight distribution](weight/policy/linear_1_w_hist.png)

#### linear_b

- mean: 0.001265
- std: 0.004200
- min: -0.008232
- max: 0.012183
- median: 0.000931

![policy linear_b weight distribution](weight/policy/linear_b_hist.png)

#### linear_w

- mean: -0.001975
- std: 0.220006
- min: -0.498047
- max: 0.502897
- median: -0.006797

![policy linear_w weight distribution](weight/policy/linear_w_hist.png)

### q1 Network

#### linear_1_b

- mean: -0.000200
- std: 0.003931
- min: -0.016112
- max: 0.018345
- median: -0.000095

![q1 linear_1_b weight distribution](weight/q1/linear_1_b_hist.png)

#### linear_1_w

- mean: -0.000619
- std: 0.055257
- min: -0.157188
- max: 0.153968
- median: -0.000640

![q1 linear_1_w weight distribution](weight/q1/linear_1_w_hist.png)

#### linear_b

- mean: 0.000538
- std: 0.005969
- min: -0.023562
- max: 0.022238
- median: 0.000358

![q1 linear_b weight distribution](weight/q1/linear_b_hist.png)

#### linear_w

- mean: -0.004350
- std: 0.186274
- min: -0.508394
- max: 0.489346
- median: -0.005139

![q1 linear_w weight distribution](weight/q1/linear_w_hist.png)

### q2 Network

#### linear_1_b

- mean: 0.000125
- std: 0.003649
- min: -0.019949
- max: 0.013071
- median: -0.000038

![q2 linear_1_b weight distribution](weight/q2/linear_1_b_hist.png)

#### linear_1_w

- mean: -0.000418
- std: 0.055168
- min: -0.156741
- max: 0.159778
- median: -0.000460

![q2 linear_1_w weight distribution](weight/q2/linear_1_w_hist.png)

#### linear_b

- mean: 0.000568
- std: 0.005718
- min: -0.017765
- max: 0.022110
- median: 0.000258

![q2 linear_b weight distribution](weight/q2/linear_b_hist.png)

#### linear_w

- mean: -0.002086
- std: 0.187244
- min: -0.496034
- max: 0.500514
- median: -0.004842

![q2 linear_w weight distribution](weight/q2/linear_w_hist.png)

## Conclusions

1. Weight distribution characteristics
2. Training stability assessment
3. Cross-network differences
