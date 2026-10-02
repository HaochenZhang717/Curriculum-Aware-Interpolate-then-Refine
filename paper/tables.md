# Paper results

Tables from [arXiv:2608.21207v1](https://arxiv.org/abs/2608.21207v1). Bold indicates the best value; underlining indicates the second best.

## Table 1

Table 1: Realistic missingness imputation on AI-READI CGM (all 352 test participants; RMSE mg/dL averaged over six missingness rates, lower is better). CAIR is best under all three mechanisms and its margin grows with difficulty. The two generic learned imputers fall below every non-constant baseline throughout. Bold = best, underline = second best, per column.

<table>
<tr><td>Method</td><td>MCAR↓</td><td>MAR↓</td><td>NMAR↓</td></tr>
<tr><td><strong>CAIR (Ours)</strong></td><td><strong>2.66</strong></td><td><strong>10.33</strong></td><td><strong>23.50</strong></td></tr>
<tr><td>linear interp</td><td><ins>2.91</ins></td><td><ins>12.34</ins></td><td><ins>28.94</ins></td></tr>
<tr><td>MICE</td><td>3.26</td><td>19.83</td><td>49.35</td></tr>
<tr><td>missForest</td><td>3.33</td><td>16.54</td><td>40.90</td></tr>
<tr><td>hot-deck</td><td>5.04</td><td>16.06</td><td>43.44</td></tr>
<tr><td>kNN</td><td>5.19</td><td>15.45</td><td>43.54</td></tr>
<tr><td>LOCF</td><td>6.22</td><td>19.84</td><td>34.44</td></tr>
<tr><td>Fourier</td><td>9.91</td><td>18.47</td><td>32.72</td></tr>
<tr><td>mean</td><td>28.59</td><td>28.81</td><td>54.86</td></tr>
<tr><td>GP-VAE</td><td>26.46</td><td>40.49</td><td>70.39</td></tr>
<tr><td>M-RNN</td><td>44.39</td><td>44.34</td><td>64.63</td></tr>
</table>

## Table 2

Table 2: Imputation performance vs. gap length on AI-READI CGM (all 352 test participants; RMSE mg/dL). CAIR is best on the mean and at 45 - 60 min, and second by a small margin at 15 - 30 min. Bold = best, underline = second, per column.

<table>
<tr><td>Method</td><td>15 min</td><td>30 min</td><td>45 min</td><td>60 min</td><td><strong>mean</strong>↓</td></tr>
<tr><td><strong>CAIR (Ours)</strong></td><td><ins>2.84</ins></td><td><ins>5.03</ins></td><td><strong>6.66</strong></td><td><strong>8.23</strong></td><td><strong>5.69</strong></td></tr>
<tr><td>akima</td><td><strong>2.76</strong></td><td><strong>5.01</strong></td><td><ins>6.87</ins></td><td>8.61</td><td><ins>5.81</ins></td></tr>
<tr><td>PCHIP</td><td>2.84</td><td>5.14</td><td>6.92</td><td><ins>8.58</ins></td><td>5.87</td></tr>
<tr><td>linear</td><td>3.20</td><td>5.64</td><td>7.45</td><td>9.09</td><td>6.34</td></tr>
<tr><td>SAITS</td><td>4.60</td><td>7.54</td><td>9.76</td><td>11.79</td><td>8.42</td></tr>
<tr><td>BRITS</td><td>6.25</td><td>9.42</td><td>11.67</td><td>13.33</td><td>10.17</td></tr>
</table>

## Table 3

Table 3: Imputation of physiologically-significant missingness on AI-READI (all 352 test participants; RMSE mg/dL, lower is better). CAIR attains the best average over the full baseline suite, and the gain concentrates in the long sleep blocks. The four weakest classical fills are in the supplement. Bold = best, underline = second best, per column.

<table>
<tr><td>Method</td><td>meal</td><td>sleep</td><td>asc</td><td>dip</td><td>comb</td><td><strong>AVG</strong>↓</td></tr>
<tr><td><strong>CAIR (Ours)</strong></td><td><strong>15.25</strong></td><td><strong>24.94</strong></td><td>9.01</td><td>7.49</td><td><strong>16.37</strong></td><td><strong>14.61</strong></td></tr>
<tr><td>PCHIP</td><td><ins>16.56</ins></td><td><ins>29.26</ins></td><td><strong>7.86</strong></td><td>7.36</td><td>18.65</td><td><ins>15.94</ins></td></tr>
<tr><td>linear</td><td>16.97</td><td>29.27</td><td>8.42</td><td><ins>7.31</ins></td><td><ins>18.30</ins></td><td>16.05</td></tr>
<tr><td>akima</td><td>17.82</td><td>36.73</td><td><ins>8.02</ins></td><td><strong>7.24</strong></td><td>19.64</td><td>17.89</td></tr>
<tr><td>Kalman</td><td>17.97</td><td>34.70</td><td>8.88</td><td>8.09</td><td>20.29</td><td>17.99</td></tr>
<tr><td>Savitzky-Golay</td><td>17.27</td><td>29.45</td><td>13.41</td><td>11.22</td><td>19.43</td><td>18.15</td></tr>
<tr><td>SAITS</td><td>21.27</td><td>32.02</td><td>11.22</td><td>10.22</td><td>19.70</td><td>18.89</td></tr>
<tr><td>BRITS</td><td>32.72</td><td>43.61</td><td>24.25</td><td>20.81</td><td>31.45</td><td>30.57</td></tr>
</table>

## Table 4

Table 4: Two analyses behind the MIMIC-III results (ABP). (a) The identical architecture, trained on the same ABP data, is less accurate than linear under MCAR and NMAR when trained on masks carried over from glucose physiology, and more accurate under all three under the random-gap curriculum. (b) Linear is worse than mean-fill ( MRR < 0 ) under MCAR/MAR; CAIR and k NN recover most of the burden. Bold = best, underline = second, per column. A third analysis, in which no imputer separates from any other when predicting mortality from arterial pressure alone, is reported in Sec. 4.7 .

<table>
<tr><td colspan="4"><em>(a) Physiological vs. random-gap training masks</em></td></tr>
<tr><td>Training masks</td><td colspan="3">ABP RMSE↓</td></tr>
<tr><td></td><td>MCAR</td><td>MAR</td><td>NMAR</td></tr>
<tr><td>linear</td><td><ins>4.40</ins></td><td>9.23</td><td><ins>11.85</ins></td></tr>
<tr><td>CAIR, CGM masks</td><td>5.47</td><td><ins>8.74</ins></td><td>12.10</td></tr>
<tr><td><strong>CAIR, random-gap</strong></td><td><strong>4.30</strong></td><td><strong>8.60</strong></td><td><strong>11.42</strong></td></tr>
<tr><td colspan="4"><em>(b) Clinical-burden recovery</em></td></tr>
<tr><td>Method</td><td colspan="3">MRR↑</td></tr>
<tr><td></td><td>MCAR</td><td>MAR</td><td>NMAR</td></tr>
<tr><td>linear</td><td>−0.31</td><td>−0.30</td><td>+0.14</td></tr>
<tr><td>kNN</td><td>+0.79</td><td>+0.74</td><td>+0.65</td></tr>
<tr><td><strong>CAIR (ours)</strong></td><td><ins>+0.71</ins></td><td><ins>+0.64</ins></td><td><ins>+0.64</ins></td></tr>
</table>

## Table 5

Table 5: Cross-modal conditioning is what transfers (AI-READI, gap-length protocol, RMSE in native units, lower is better). Column headings are gap lengths in minutes; “Resp.” is respiration and “AR (bidir.)” is bidirectional AR. Univariate CAIR is less accurate than linear at every gap length on both signals; adding the co-recorded context makes it best everywhere. On glucose the same context adds nothing (Sec. 4.5 ). Bold = best, underline = second, per column.

<table>
<tr><td></td><td colspan="4">Heart rate (bpm)↓</td><td colspan="4">Resp. (br/min)↓</td></tr>
<tr><td>Method</td><td>15</td><td>30</td><td>45</td><td>60</td><td>15</td><td>30</td><td>45</td><td>60</td></tr>
<tr><td>Akima</td><td>4.54</td><td>5.93</td><td>6.86</td><td>7.87</td><td>2.73</td><td>3.23</td><td>3.72</td><td>4.30</td></tr>
<tr><td>AR (bidir.)</td><td>4.71</td><td>5.62</td><td><ins>6.09</ins></td><td><ins>6.72</ins></td><td>2.77</td><td>3.09</td><td><ins>3.33</ins></td><td><ins>3.57</ins></td></tr>
<tr><td>PCHIP</td><td>4.45</td><td>5.62</td><td>6.27</td><td>7.05</td><td>2.63</td><td>3.07</td><td>3.43</td><td>3.76</td></tr>
<tr><td>Linear</td><td><ins>4.40</ins></td><td><ins>5.52</ins></td><td>6.12</td><td>6.86</td><td><ins>2.61</ins></td><td><ins>3.00</ins></td><td>3.34</td><td>3.65</td></tr>
<tr><td colspan="9">CAIR (ours)</td></tr>
<tr><td>univariate</td><td>4.96</td><td>6.49</td><td>6.74</td><td>7.51</td><td>2.90</td><td>3.92</td><td>4.50</td><td>5.21</td></tr>
<tr><td><strong>multivariate</strong></td><td><strong>3.50</strong></td><td><strong>4.19</strong></td><td><strong>4.29</strong></td><td><strong>4.43</strong></td><td><strong>1.88</strong></td><td><strong>2.54</strong></td><td><strong>2.62</strong></td><td><strong>2.72</strong></td></tr>
</table>

## Table 6

Table 6: Cross-domain transfer to MIMIC-III ICU vitals (reconstruction RMSE, mean over six missingness rates, lower is better). CAIR is the most accurate method on arterial pressure under every mechanism; heart rate is a boundary case. The CAIR row is the multivariate variant; both variants, the remaining baselines and per-mechanism Wilcoxon tests are in the supplement. Bold = best, underline = second, per column.

<table>
<tr><td></td><td colspan="3">ABP-mean (mmHg)↓</td><td colspan="3">HR (bpm)↓</td></tr>
<tr><td>Method</td><td>MCAR</td><td>MAR</td><td>NMAR</td><td>MCAR</td><td>MAR</td><td>NMAR</td></tr>
<tr><td><strong>CAIR (Ours)</strong></td><td><strong>4.30</strong></td><td><strong>8.60</strong></td><td><strong>11.42</strong></td><td><ins>2.26</ins></td><td><strong>5.28</strong></td><td><ins>6.48</ins></td></tr>
<tr><td>linear interp</td><td><ins>4.40</ins></td><td><ins>9.23</ins></td><td><ins>11.85</ins></td><td><strong>2.22</strong></td><td><ins>5.50</ins></td><td><strong>6.42</strong></td></tr>
<tr><td>GP-VAE</td><td>11.03</td><td>12.82</td><td>17.94</td><td>9.17</td><td>14.01</td><td>18.00</td></tr>
<tr><td>M-RNN</td><td>13.45</td><td>13.19</td><td>18.35</td><td>15.28</td><td>14.53</td><td>18.60</td></tr>
</table>

## Table 7

Table 7: Component ablation on AI-READI CGM, on the protocol of Table 1 (all 352 test participants; RMSE mg/dL over six missingness rates). ALL is the mean of the three mechanisms, Δ the change against the full model. Rows are single training runs under the published recipe, differing only in the ablated component; linear is repeated from Table 1 for scale. Bold = best, underline = second best, per metric column.

<table>
<tr><td>Variant</td><td>MCAR</td><td>MAR</td><td>NMAR</td><td><strong>ALL</strong>↓</td><td>Δ</td></tr>
<tr><td><strong>CAIR (full)</strong></td><td><strong>2.66</strong></td><td><ins>10.33</ins></td><td><strong>23.50</strong></td><td><strong>12.16</strong></td><td>n/a</td></tr>
<tr><td>w/o aux. loss</td><td>6.36</td><td>10.75</td><td>27.59</td><td>14.90</td><td>+2.74</td></tr>
<tr><td>w/o residual</td><td>4.45</td><td>10.34</td><td>27.38</td><td>14.06</td><td>+1.90</td></tr>
<tr><td>GRU→self-attn.</td><td>4.15</td><td><strong>10.22</strong></td><td><ins>27.14</ins></td><td><ins>13.84</ins></td><td>+1.68</td></tr>
<tr><td>linear interp</td><td><ins>2.91</ins></td><td>12.34</td><td>28.94</td><td>14.73</td><td>+2.57</td></tr>
</table>

## Table 8

Table 8: External-cohort pretraining does not improve accuracy (physiological protocol, 352 participants, single seed; RMSE mg/dL). Every external cell is within 0.2 mg/dL of the control and none improves on it. Bold = best, underline = second best, per column.

<table>
<tr><td>Pretrain, then fine-tune</td><td>meal</td><td>sleep</td><td>asc</td><td>dip</td><td>comb</td><td>AVG</td></tr>
<tr><td>control (AI-READI only)</td><td><strong>15.60</strong></td><td>27.52</td><td><strong>5.96</strong></td><td><strong>5.14</strong></td><td><strong>14.74</strong></td><td><strong>13.79</strong></td></tr>
<tr><td>+HUPA-UCM (T1D)</td><td>15.99</td><td><strong>26.32</strong></td><td>6.21</td><td>5.36</td><td>15.66</td><td>13.91</td></tr>
<tr><td>+OhioT1DM (T1D)</td><td>15.77</td><td><ins>27.14</ins></td><td><ins>6.08</ins></td><td>5.21</td><td>15.25</td><td><ins>13.89</ins></td></tr>
<tr><td>+Shanghai (T2D)</td><td><ins>15.71</ins></td><td>27.58</td><td>6.13</td><td><ins>5.20</ins></td><td><ins>15.18</ins></td><td>13.96</td></tr>
</table>

## Table 9

Table 9: The four baselines omitted from Table 3 of the main paper (AI-READI, physiological five-strategy protocol, all 352 test participants; RMSE mg/dL, lower is better). The lower block is the omitted group; CAIR and Savitzky-Golay are repeated from Table 3 as the best method and the weakest one carried there. Every omitted method is at least 4.9 mg/dL behind that floor on the average. Bold = best, underline = second best, per column.

<table>
<tr><td>Method</td><td>meal</td><td>sleep</td><td>asc</td><td>dip</td><td>comb</td><td><strong>AVG</strong>↓</td></tr>
<tr><td><strong>CAIR (Ours)</strong></td><td><strong>15.25</strong></td><td><strong>24.94</strong></td><td><strong>9.01</strong></td><td><strong>7.49</strong></td><td><strong>16.37</strong></td><td><strong>14.61</strong></td></tr>
<tr><td>Savitzky-Golay</td><td><ins>17.27</ins></td><td><ins>29.45</ins></td><td><ins>13.41</ins></td><td><ins>11.22</ins></td><td><ins>19.43</ins></td><td><ins>18.15</ins></td></tr>
<tr><td>EWMA</td><td>20.59</td><td>31.69</td><td>21.45</td><td>18.32</td><td>23.11</td><td>23.03</td></tr>
<tr><td>local mean</td><td>22.66</td><td>32.23</td><td>24.00</td><td>20.65</td><td>24.40</td><td>24.79</td></tr>
<tr><td>cubic spline</td><td>22.64</td><td>45.22</td><td>21.90</td><td>18.62</td><td>23.85</td><td>26.44</td></tr>
<tr><td>LOCF</td><td>25.91</td><td>37.33</td><td>34.19</td><td>29.36</td><td>31.81</td><td>31.72</td></tr>
</table>

## Table 10

Table 10: Conditioning-modality ladder on AI-READI CGM (five-seed ensemble per rung; RMSE mg/dL). Adding modalities does not improve glucose imputation; the AVG spread ( 12.98 - 13.14 ) is within seed noise. AVG is over all five physiological strategies; the combined column is omitted for space. Absolute values are not comparable to Table 3 of the main paper. Bold / underline mark the best/second AVG.

<table>
<tr><td>Conditioning set</td><td>meal</td><td>sleep</td><td>asc</td><td>dip</td><td><strong>AVG</strong></td></tr>
<tr><td>target only (control)</td><td>14.94</td><td>25.17</td><td>6.07</td><td>4.93</td><td>13.11</td></tr>
<tr><td>+heart rate</td><td>14.93</td><td>25.16</td><td>6.03</td><td>4.92</td><td>13.10</td></tr>
<tr><td>+steps, calories</td><td>14.89</td><td>25.25</td><td>6.10</td><td>4.97</td><td>13.14</td></tr>
<tr><td>+sleep state</td><td>14.89</td><td>25.27</td><td>5.98</td><td>4.92</td><td>13.09</td></tr>
<tr><td>+respiration, stress</td><td>14.79</td><td>24.96</td><td>6.00</td><td>4.89</td><td><strong>12.98</strong></td></tr>
<tr><td>+environment</td><td>14.82</td><td>24.83</td><td>6.24</td><td>5.06</td><td>13.10</td></tr>
<tr><td>+clinical</td><td>14.79</td><td>24.85</td><td>6.18</td><td>5.01</td><td><ins>13.06</ins></td></tr>
<tr><td>+ECG</td><td>14.86</td><td>24.87</td><td>6.26</td><td>5.09</td><td>13.14</td></tr>
<tr><td>+retinal</td><td>14.86</td><td>24.90</td><td>6.12</td><td>4.98</td><td>13.07</td></tr>
</table>

## Table 11

Table 11: Complete MIMIC-III arterial-pressure reconstruction RMSE (mmHg, mean over six missingness rates). Bold = best, underline = second best, per column.

<table>
<tr><td>Method</td><td>MCAR</td><td>MAR</td><td>NMAR</td></tr>
<tr><td>CAIR (univariate)</td><td><strong>4.26</strong></td><td><strong>8.58</strong></td><td><ins>11.44</ins></td></tr>
<tr><td>CAIR (multivariate)</td><td><ins>4.30</ins></td><td><ins>8.60</ins></td><td><strong>11.42</strong></td></tr>
<tr><td>linear interp</td><td>4.40</td><td>9.23</td><td>11.85</td></tr>
<tr><td>missForest</td><td>4.62</td><td>9.60</td><td>12.91</td></tr>
<tr><td>MICE</td><td>4.88</td><td>9.02</td><td>12.65</td></tr>
<tr><td>kNN</td><td>5.27</td><td>8.91</td><td>12.69</td></tr>
<tr><td>hot-deck</td><td>5.82</td><td>9.11</td><td>12.81</td></tr>
<tr><td>LOCF</td><td>5.89</td><td>10.43</td><td>12.79</td></tr>
<tr><td>Fourier</td><td>5.94</td><td>11.28</td><td>13.56</td></tr>
<tr><td>mean</td><td>9.33</td><td>9.93</td><td>14.01</td></tr>
<tr><td>mode</td><td>11.42</td><td>11.82</td><td>15.11</td></tr>
<tr><td>GP-VAE</td><td>11.03</td><td>12.82</td><td>17.94</td></tr>
<tr><td>M-RNN</td><td>13.45</td><td>13.19</td><td>18.35</td></tr>
</table>

## Table 12

Table 12: Complete MIMIC-III heart-rate reconstruction RMSE (bpm, mean over six missingness rates). The CAIR (multivariate) and linear rows are those of Table 6 of the main paper. CAIR is the most accurate method under MAR and linear interpolation under MCAR and NMAR, the two separated by at most 0.22 bpm; the per-mechanism significance tests for those differences are in Table 13 . Bold = best, underline = second best, per column.

<table>
<tr><td>Method</td><td>MCAR</td><td>MAR</td><td>NMAR</td></tr>
<tr><td>CAIR (univariate)</td><td><ins>2.26</ins></td><td><ins>5.36</ins></td><td>6.72</td></tr>
<tr><td>CAIR (multivariate)</td><td><ins>2.26</ins></td><td><strong>5.28</strong></td><td><ins>6.48</ins></td></tr>
<tr><td>linear interp</td><td><strong>2.22</strong></td><td>5.50</td><td><strong>6.42</strong></td></tr>
<tr><td>missForest</td><td>2.37</td><td>7.40</td><td>9.25</td></tr>
<tr><td>MICE</td><td>2.49</td><td>7.55</td><td>9.63</td></tr>
<tr><td>kNN</td><td>2.69</td><td>7.09</td><td>9.21</td></tr>
<tr><td>hot-deck</td><td>2.90</td><td>7.19</td><td>9.28</td></tr>
<tr><td>LOCF</td><td>3.07</td><td>7.06</td><td>7.79</td></tr>
<tr><td>Fourier</td><td>3.31</td><td>7.17</td><td>7.86</td></tr>
<tr><td>mean</td><td>8.41</td><td>9.22</td><td>11.69</td></tr>
<tr><td>mode</td><td>10.88</td><td>10.39</td><td>14.10</td></tr>
<tr><td>GP-VAE</td><td>9.17</td><td>14.01</td><td>18.00</td></tr>
<tr><td>M-RNN</td><td>15.28</td><td>14.53</td><td>18.60</td></tr>
</table>

## Table 13

Table 13: Paired Wilcoxon signed-rank tests on MIMIC-III , CAIR (multivariate) versus linear interpolation, per (window, rate) pair. Δ is the mean of the per-pair RMSE differences in native units, which need not equal the difference of the aggregate RMSEs in Table 6 of the main paper because RMSE does not aggregate linearly; negative favors CAIR. All three arterial-pressure tests favor CAIR; on heart rate, linear interpolation is significantly better under MCAR and the remaining two differences are not significant ( α = 0.05 ).

<table>
<tr><td>Signal</td><td>Mechanism</td><td>Δ</td><td>p</td></tr>
<tr><td rowspan="3">ABP-mean (mmHg)</td><td>MCAR</td><td>−0.105</td><td>7.9×10−9</td></tr>
<tr><td>MAR</td><td>−0.625</td><td>4.1×10−21</td></tr>
<tr><td>NMAR</td><td>−0.375</td><td>2.9×10−3</td></tr>
<tr><td rowspan="3">Heart rate (bpm)</td><td>MCAR</td><td>+0.040</td><td>7.0×10−40</td></tr>
<tr><td>MAR</td><td>−0.220</td><td>0.83</td></tr>
<tr><td>NMAR</td><td>+0.060</td><td>0.37</td></tr>
</table>

## Table 14

Table 14: Only CAIR ranks among the best on both axes (MIMIC-III ABP, NMAR, mean over six rates). Reconstruction RMSE (mmHg, lower better) and clinical-burden recovery (MRR on time in/above/below range, higher better). Interpolants match the RMSE of CAIR but fail to preserve the burden; tabular and neural imputers recover the burden at much higher RMSE, with k NN and hot-deck matching its recovery at ∼ 11 % higher RMSE. Bold = best, underline = second best, per column.

<table>
<tr><td>Method</td><td>RMSE (mmHg)↓</td><td>Burden MRR↑</td></tr>
<tr><td colspan="3"><em>Low RMSE, burden not preserved</em></td></tr>
<tr><td>linear interp</td><td>11.85</td><td>+0.14</td></tr>
<tr><td>LOCF</td><td>12.79</td><td>+0.07</td></tr>
<tr><td>Fourier</td><td>13.56</td><td>+0.05</td></tr>
<tr><td colspan="3"><em>Recovers burden, high RMSE</em></td></tr>
<tr><td>MICE</td><td>12.65</td><td>+0.63</td></tr>
<tr><td>kNN</td><td>12.69</td><td>+0.65</td></tr>
<tr><td>hot-deck</td><td>12.81</td><td>+0.65</td></tr>
<tr><td>GP-VAE</td><td>17.94</td><td>+0.64</td></tr>
<tr><td>M-RNN</td><td>18.35</td><td>+0.64</td></tr>
<tr><td colspan="3"><em>Both</em></td></tr>
<tr><td>CAIR (univariate)</td><td><ins>11.44</ins></td><td>+0.65</td></tr>
<tr><td><strong>CAIR (multivar.)</strong></td><td><strong>11.42</strong></td><td>+0.64</td></tr>
</table>

## Table 15

Table 15: Hard outcomes do not reward reconstruction fidelity (MIMIC-III, in-hospital mortality predicted from arterial pressure alone, 30 % missingness, 1D-CNN readout; n = 1200 under MCAR and 1181 under NMAR). The oracle is the recorded trace and is the ceiling for this task, yet four fills exceed it under MCAR and all five do under NMAR, because interpolation denoises the vital. No column is ordered as reconstruction accuracy would predict, and no best value is marked.

<table>
<tr><td>Fill</td><td>MCAR AUROC</td><td>NMAR AUROC</td></tr>
<tr><td>oracle (recorded trace)</td><td>0.679</td><td>0.678</td></tr>
<tr><td>mean-fill</td><td>0.677</td><td>0.683</td></tr>
<tr><td>kNN</td><td>0.680</td><td>0.685</td></tr>
<tr><td>CAIR (multivar.)</td><td>0.688</td><td>0.688</td></tr>
<tr><td>linear interp</td><td>0.692</td><td>0.688</td></tr>
<tr><td>PCHIP</td><td>0.692</td><td>0.689</td></tr>
</table>

## Table 16

Table 16: The CGM result holds in every diabetes study group (AI-READI, NMAR, all 352 test participants). CAIR attains lower reconstruction error and higher time-in-range recovery than linear interpolation in all four groups. Bold = better of the two, per group and metric.

<table>
<tr><td></td><td colspan="2">RMSE (mg/dL)↓</td><td colspan="2">TIR recovery↑</td></tr>
<tr><td>Study group</td><td>CAIR</td><td>linear</td><td>CAIR</td><td>linear</td></tr>
<tr><td>healthy</td><td><strong>18.29</strong></td><td>21.49</td><td><strong>0.165</strong></td><td>0.097</td></tr>
<tr><td>pre-diabetes</td><td><strong>15.90</strong></td><td>21.15</td><td><strong>0.467</strong></td><td>0.382</td></tr>
<tr><td>oral medication</td><td><strong>33.96</strong></td><td>40.19</td><td><strong>0.601</strong></td><td>0.556</td></tr>
<tr><td>insulin-dependent</td><td><strong>27.23</strong></td><td>35.17</td><td><strong>0.608</strong></td><td>0.489</td></tr>
</table>
