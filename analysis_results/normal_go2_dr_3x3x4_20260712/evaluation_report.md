# Go2 Normal DR 3x3x4 Evaluation Summary

## Ranking Method

Deployment robustness is ranked on V1-V3 using 35% mean survival, 25% worst-domain survival, 20% domain-normalized return, and 20% tracking quality. V0 clean is retained as a sanity check and is not used to inflate the robustness score.

## Top 10

|Rank|Job|Expert DR|Dataset DR|Interface|Score|Mean survival|Worst survival|Mean return|Tracking error|
|---:|---|---|---|---|---:|---:|---:|---:|---:|
|1|E0D0P2|default|default|obs_noise|70.35|0.783|0.445|107.09|0.0878|
|2|E0D0P3|default|default|action_obs_noise|68.79|0.794|0.383|109.86|0.0919|
|3|E1D1P2|friend_half|friend_half|obs_noise|57.51|0.588|0.369|79.65|0.0825|
|4|E2D1P0|friend_full|friend_half|none|54.61|0.682|0.111|94.83|0.0752|
|5|E0D0P0|default|default|none|53.65|0.697|0.092|96.61|0.0917|
|6|E2D0P2|friend_full|default|obs_noise|51.99|0.683|0.049|96.35|0.0869|
|7|E1D0P2|friend_half|default|obs_noise|51.20|0.601|0.204|79.16|0.0958|
|8|E1D1P3|friend_half|friend_half|action_obs_noise|51.00|0.534|0.257|72.52|0.0881|
|9|E1D0P0|friend_half|default|none|50.93|0.682|0.045|94.18|0.0943|
|10|E0D0P1|default|default|action_noise|50.29|0.676|0.028|93.75|0.0915|

## Raw Evaluation Rewards

`V0` is the clean MJLab environment. `V1`, `V2`, and `V3` use default, friend-half, and friend-full DR respectively. Mean DR reward is the arithmetic mean of V1-V3 and does not include V0. The robust score is a separate 0-100 composite metric, not a reward.

|Rank|Job|Robust score|V0 clean|V1 default|V2 half|V3 full|Mean DR reward|
|---:|---|---:|---:|---:|---:|---:|---:|
|1|E0D0P2|70.35|142.53|124.03|141.30|55.94|107.09|
|2|E0D0P3|68.79|142.24|141.31|141.13|47.13|109.86|
|3|E1D1P2|57.51|144.79|46.60|143.52|48.84|79.65|
|4|E2D1P0|54.61|145.82|129.74|144.86|9.88|94.83|
|5|E0D0P0|53.65|142.50|141.78|141.72|6.32|96.61|
|6|E2D0P2|51.99|144.99|144.20|143.86|0.98|96.35|
|7|E1D0P2|51.20|141.00|76.60|139.82|21.07|79.16|
|8|E1D1P3|51.00|145.16|31.14|144.28|42.14|72.52|
|9|E1D0P0|50.93|143.02|142.02|141.50|-0.96|94.18|
|10|E0D0P1|50.29|143.22|142.35|142.29|-3.37|93.75|
|11|E1D0P1|49.37|141.95|141.22|140.83|-4.20|92.62|
|12|E0D1P2|49.16|144.80|41.22|143.41|26.73|70.45|
|13|E1D1P0|46.14|144.87|54.71|143.61|13.46|70.59|
|14|E0D2P2|45.23|140.92|42.62|140.18|18.92|67.24|
|15|E2D1P1|43.81|145.43|31.49|144.73|14.06|63.43|
|16|E1D2P1|43.14|138.68|21.57|138.05|26.13|61.92|
|17|E2D2P1|40.04|143.69|13.75|143.31|14.05|57.04|
|18|E0D2P1|39.42|31.45|25.56|74.91|35.98|45.48|
|19|E2D2P0|39.04|142.87|11.89|142.21|11.99|55.36|
|20|E1D0P3|38.74|141.91|15.04|140.73|8.94|54.90|
|21|E0D1P0|38.12|144.60|16.11|143.39|4.88|54.79|
|22|E1D2P0|37.33|139.83|27.06|138.82|5.09|56.99|
|23|E0D1P1|29.28|144.86|37.17|19.77|16.54|24.49|
|24|E0D1P3|29.08|144.81|31.81|15.68|23.27|23.59|
|25|E1D1P1|28.30|145.01|42.33|9.77|14.49|22.20|
|26|E0D2P3|27.14|13.98|27.84|21.63|17.87|22.45|
|27|E0D2P0|25.35|31.30|18.19|25.68|11.76|18.55|
|28|E2D1P2|24.59|145.50|17.45|19.28|11.25|15.99|
|29|E2D0P1|24.26|16.07|43.09|13.03|5.57|20.56|
|30|E2D2P2|22.69|143.18|11.18|20.35|9.12|13.55|
|31|E2D1P3|21.04|145.27|9.44|14.02|6.24|9.90|
|32|E2D2P3|20.42|15.43|9.45|11.82|8.02|9.76|
|33|E2D0P0|20.36|145.13|15.09|10.45|5.48|10.34|
|34|E2D0P3|20.27|145.16|6.33|16.63|5.55|9.50|
|35|E1D2P3|19.61|34.31|14.15|20.14|13.89|16.06|
|36|E1D2P2|12.96|7.98|7.64|7.96|8.06|7.88|

## Factor Means

|Factor|Level|N|Score|Mean survival|Worst survival|Mean return|Tracking error|
|---|---|---:|---:|---:|---:|---:|---:|
|expert_dr|default|12|43.82|0.466|0.193|61.20|0.0929|
|expert_dr|friend_full|12|31.93|0.299|0.101|38.05|0.0857|
|expert_dr|friend_half|12|40.52|0.454|0.153|59.06|0.1209|
|dataset_dr|default|12|45.85|0.534|0.134|72.08|0.0909|
|dataset_dr|friend_full|12|31.03|0.299|0.146|36.02|0.1280|
|dataset_dr|friend_half|12|39.39|0.386|0.167|50.20|0.0806|
|interface_noise|action_noise|9|38.66|0.413|0.127|53.50|0.0934|
|interface_noise|action_obs_noise|9|32.90|0.297|0.167|36.50|0.1066|
|interface_noise|none|9|40.62|0.461|0.099|61.36|0.0899|
|interface_noise|obs_noise|9|42.85|0.454|0.203|59.71|0.1093|

## Top Candidate Checkpoints

- E0D0P2: /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712/policies/E0/D0/P2/run/step48828
- E0D0P3: /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712/policies/E0/D0/P3/run/step48828
- E1D1P2: /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712/policies/E1/D1/P2/run/step48828
- E2D1P0: /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712/policies/E2/D1/P0/run/step48828
- E0D0P0: /data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712/policies/E0/D0/P0/run/step48828
