Rates and intervals use the same source clusters across fitted models. Brackets give 95% percentile intervals conditional on the fitted adapters. Mixture rows average seeds 182 and 183.

### Same-question protocol comparison

| Non-tool share | Gen / MCQ required recall (%) | MCQ − gen (pp) | Gen / MCQ false calls (%) | MCQ − gen (pp) | Gen / MCQ accuracy (%) | MCQ − gen (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NF4 base | 99.0 / 100.0 | 1.0 [0.0, 3.3] | 47.5 / 87.0 | 39.5 [31.8, 46.9] | 59.0 / 37.3 | −21.7 [−26.8, −16.6] |
| 0% | 100.0 / 100.0 | 0.0 [0.0, 0.0] | 91.5 / 92.0 | 0.5 [−2.5, 3.7] | 36.3 / 35.0 | −1.3 [−3.8, 1.0] |
| 10% | 96.0 / 100.0 | 4.0 [1.0, 7.5] | 37.8 / 65.0 | 27.3 [19.8, 34.8] | 68.3 / 54.2 | −14.2 [−19.7, −8.8] |
| 25% | 83.0 / 100.0 | 17.0 [10.0, 23.9] | 18.8 / 56.0 | 37.3 [29.5, 44.9] | 73.7 / 60.0 | −13.7 [−19.9, −7.5] |
| 50% | 67.0 / 99.0 | 32.0 [23.0, 40.7] | 10.0 / 47.0 | 37.0 [29.4, 44.2] | 74.0 / 65.5 | −8.5 [−15.4, −1.7] |

### Protocol accuracy and raw-sum sensitivity

| Model · seed | Generation correct / all | Normalized MCQ correct / all | Raw MCQ correct / all | Raw MCQ tool / required | Raw MCQ tool / non-tool |
| --- | ---: | ---: | ---: | ---: | ---: |
| NF4 base | 177/300 | 112/300 | 125/300 | 100/100 | 146/200 |
| 0% · 182 | 109/300 | 105/300 | 110/300 | 99/100 | 171/200 |
| 0% · 183 | 109/300 | 105/300 | 111/300 | 100/100 | 170/200 |
| 10% · 182 | 208/300 | 165/300 | 174/300 | 98/100 | 106/200 |
| 10% · 183 | 202/300 | 160/300 | 172/300 | 97/100 | 104/200 |
| 25% · 182 | 219/300 | 186/300 | 201/300 | 98/100 | 77/200 |
| 25% · 183 | 223/300 | 174/300 | 189/300 | 97/100 | 87/200 |
| 50% · 182 | 222/300 | 203/300 | 214/300 | 95/100 | 62/200 |
| 50% · 183 | 222/300 | 190/300 | 199/300 | 94/100 | 72/200 |

### Required calls (100)

| Model · seed | MCQ selects | Generation calls | Generation does not call |
| --- | ---: | ---: | ---: |
| NF4 base | Tool | 99 | 1 |
| NF4 base | Text | 0 | 0 |
| 0% · 182 | Tool | 100 | 0 |
| 0% · 182 | Text | 0 | 0 |
| 0% · 183 | Tool | 100 | 0 |
| 0% · 183 | Text | 0 | 0 |
| 10% · 182 | Tool | 95 | 5 |
| 10% · 182 | Text | 0 | 0 |
| 10% · 183 | Tool | 97 | 3 |
| 10% · 183 | Text | 0 | 0 |
| 25% · 182 | Tool | 80 | 20 |
| 25% · 182 | Text | 0 | 0 |
| 25% · 183 | Tool | 86 | 14 |
| 25% · 183 | Text | 0 | 0 |
| 50% · 182 | Tool | 64 | 36 |
| 50% · 182 | Text | 0 | 0 |
| 50% · 183 | Tool | 69 | 29 |
| 50% · 183 | Text | 1 | 1 |

### Non-tool requests (200)

| Model · seed | MCQ selects | Generation calls | Generation does not call |
| --- | ---: | ---: | ---: |
| NF4 base | Tool | 90 | 84 |
| NF4 base | Text | 5 | 21 |
| 0% · 182 | Tool | 178 | 6 |
| 0% · 182 | Text | 5 | 11 |
| 0% · 183 | Tool | 178 | 6 |
| 0% · 183 | Text | 5 | 11 |
| 10% · 182 | Tool | 61 | 67 |
| 10% · 182 | Text | 13 | 59 |
| 10% · 183 | Tool | 66 | 66 |
| 10% · 183 | Text | 11 | 57 |
| 25% · 182 | Tool | 27 | 80 |
| 25% · 182 | Text | 8 | 85 |
| 25% · 183 | Tool | 31 | 86 |
| 25% · 183 | Text | 9 | 74 |
| 50% · 182 | Tool | 8 | 81 |
| 50% · 182 | Text | 8 | 103 |
| 50% · 183 | Tool | 16 | 83 |
| 50% · 183 | Text | 8 | 93 |

### Discordance on all 300 prompts

| Non-tool share | MCQ selects call, generation text (%) | Generation calls, MCQ selects text (%) |
| --- | ---: | ---: |
| NF4 base | 28.3 [23.5, 33.3] | 1.7 [0.3, 3.3] |
| 0% | 2.0 [0.7, 3.7] | 1.7 [0.3, 3.2] |
| 10% | 23.5 [19.3, 28.2] | 4.0 [2.0, 6.5] |
| 25% | 33.3 [28.5, 38.4] | 2.8 [1.2, 4.8] |
| 50% | 38.2 [32.8, 43.6] | 2.8 [1.2, 4.7] |
