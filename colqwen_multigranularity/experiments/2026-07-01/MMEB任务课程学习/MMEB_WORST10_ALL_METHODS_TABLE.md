# MMEB Worst10 All Methods Table

Rows are the original sym160 4k MMEB worst10 datasets. Values are P@1/R@1. `-` means this targeted eval scope did not include that dataset.

| Dataset | baseline q2d_sum | q2d_mean | bi_mean | global_local_bi_mean | bi_topk_mean | vqa_hard_s64 target | replay20_s64 target | compositional_s64 target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FashionIQ | 0.025 | 0.025 | 0.020 | 0.019 | 0.018 | - | - | 0.039 |
| Country211 | 0.088 | 0.088 | 0.126 | 0.122 | 0.108 | - | - | - |
| CIRR | 0.105 | 0.105 | 0.095 | 0.094 | 0.092 | - | - | 0.084 |
| InfographicsVQA | 0.137 | 0.139 | 0.238 | 0.176 | 0.099 | 0.004 | 0.004 | - |
| Visual7W | 0.147 | 0.147 | 0.291 | 0.217 | 0.170 | 0.000 | 0.000 | - |
| GQA | 0.155 | 0.155 | 0.184 | 0.157 | 0.120 | 0.008 | 0.009 | - |
| ChartQA | 0.174 | 0.174 | 0.316 | 0.247 | 0.126 | 0.002 | 0.002 | - |
| A-OKVQA | 0.182 | 0.182 | 0.256 | 0.227 | 0.139 | 0.005 | 0.005 | - |
| ScienceQA | 0.198 | 0.198 | 0.222 | 0.238 | 0.203 | 0.010 | 0.010 | - |
| OK-VQA | 0.214 | 0.214 | 0.310 | 0.270 | 0.168 | 0.005 | 0.005 | - |
| Average | 0.143 | 0.143 | 0.206 | 0.177 | 0.124 | 0.005 | 0.005 | 0.061 |
