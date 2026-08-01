# vr_calibration

## 바로 실행

orientation 오차가 크지 않을 때, 기존 `T_SA`를 유지하고 position/base calibration만 갱신:

```bash
cd <your_ros2_workspace>
source install/setup.bash
ros2 run vr_calibration vr_calibration
```

orientation 오차가 클 때, `/calibrated_pose` 기준 rotation도 다시 맞춤:

```bash
cd <your_ros2_workspace>
source install/setup.bash
ros2 run vr_calibration vr_calibration --ros-args \
  -p t_sa_mode:=update \
  -p t_sa_max_delta_deg:=180.0
```

`vr_calibration`은 UR robot EE pose와 Vive tracker raw pose를 같은 target waypoint에서 수집한 뒤, `vive_tracker_ros2` 런타임이 사용할 calibration YAML을 생성하는 ROS 2 패키지다.

이 문서는 `nrs_imitation` 전체가 아니라 `behavior_ws/src/vr_calibration` 패키지 기준으로만 정리한다.

## 입력과 출력

입력 topic:

- `/ur10skku/currentP`: `Float64MultiArray`, robot current pose `[x y z wx wy wz]`
- `/raw_pose`: `PoseStamped`, Vive tracker raw pose
- `/calibrated_pose`: `Float64MultiArray`, `T_SA` update 모드에서 현재 calibrated rotation을 읽기 위해 사용
- `/ur10skku/ftdata_tcp_raw`: `WrenchStamped`, Y2FT_AQ가 publish하는 보상 전 TCP-frame robot FT 값

주요 파일:

- `vr_calibration/txt/for_vr_calibration_point_v7.txt`: 320 mm EE-to-TCP spindle용 기본 target waypoint 파일
- `vr_calibration/txt/ur10_ee.txt`: 캡처된 EE pose 기록
- `vr_calibration/txt/ur10_vr.txt`: 캡처된 VR pose 기록
- `vive_tracker_ros2/yaml/calibration_matrix.yaml`: 최종 calibration YAML
- `nrs_ft_aq2/config/spindle_gravity.yaml`: robot FT로 식별한 공통 spindle 중력보상 행렬

## Spindle 중력보상 동시 calibration

이 모드에서는 spindle을 로봇에 장착하고 **로봇 FT만 실행**한다. 교시장치의 `nrs_ft_aq2`는 실행하지 않는다.
Y2FT_AQ는 센서 zero 이후의 값을 TCP 축으로만 회전한 `/ur10skku/ftdata_tcp_raw`를 제공하고,
VR calibration 노드는 정지한 각 capture pose의 wrench 중앙값을 함께 저장한다.

여러 자세에 대해 다음 모델을 식별한다.

```text
wrench_tcp = bias + G_spindle(6x3) * gravity_tcp
```

runtime에 저장되는 `G_spindle`은 자유 6x3 회귀값을 그대로 쓰지 않고, 식별한 질량과 3축 CoM으로 만든
물리적으로 일관된 행렬이다. STL 형상, STL density, STL CoM은 사용하지 않는다. 센서 zero로 제거된 상수값은
회귀의 `bias`가 흡수하며 runtime은 zero 자세와 현재 자세의 `delta gravity`에만 행렬을 적용한다.

기본 v7 waypoint는 `EE2TCP` 길이 320 mm를 기준으로 만든다. 처음 8개 capture pose는 수직 spindle로
중앙 작업공간의 위치 분포를 만들고, 나머지 24개는 EE를 안전한 고점 부근에 유지한 채 기존 v6에서 사용한
자세를 회전 변화가 작은 순서로 배치한다. TCP/EE 끝점 범위와 중력 방향 분포를 수치 검증했지만, 이는 로봇 및
주변 설비의 실제 collision model을 대신하지 않는다. 최초 실행은 반드시 저속/수동 정지 준비 상태에서 확인한다.

실행 순서:

```bash
# nrs_forcecon@192.168.0.151
cd /home/nrs_forcecon/dev_ws
source install/setup.bash
ros2 run Y2FT_AQ FTGetMain

# eunseop_nrs3 (Vive tracker/robot waypoint 노드가 준비된 뒤)
cd /home/eunseop/nrs_imitation/behavior_ws
source install/setup.bash
ros2 run vr_calibration vr_calibration
```

성공 조건을 모두 만족할 때만 로컬 `nrs_ft_aq2/config/spindle_gravity.yaml`을 교체하고 기존 파일은
`.bak`으로 보존한다. 기본 설정에서는 passwordless SSH로 다음 원격 파일도 같은 YAML로 교체한다.

```text
nrs_forcecon@192.168.0.151:/home/nrs_forcecon/dev_ws/src/y2_ur10skku_control/Y2FT_AQ/config/spindle_gravity.yaml
```

원격 파일도 `.bak`으로 보존된다. Y2FT_AQ는 YAML을 시작할 때 읽으므로 calibration이 끝난 뒤 로봇 FT 노드를
재시작해야 새 행렬이 적용된다. 교시장치 FT는 calibration 중 꺼져 있었으므로, 나중에 실행할 때 새 로컬 YAML을 읽는다.

주요 성공 로그:

```text
[GRAVITY_CAPTURE] ...
[GRAVITY_SAVED] n=... cond=... mass=... com=... rms=...
[GRAVITY_REMOTE] updated ...
```

`[GRAVITY_REJECTED]`가 나오면 기존 gravity YAML은 유지된다. 대표적인 거부 조건은 자세 방향 rank 부족,
condition number 초과, 비현실적인 질량/CoM, force/torque residual RMS 초과다.

기본 설정에서는 32개 gravity pose를 모두 최종 fit에 쓰지 않는다. 먼저 전체 pose로 1차 fit을 수행해
pose별 force/torque residual을 계산하고, residual이 작은 good-quality pose만 골라 다시 fit한다.
`gravity_quality_min_pose_samples`개 이상이 남고 최종 RMS/condition/mass/CoM 검사를 통과할 때만
`spindle_gravity.yaml`을 저장한다.

생성되는 YAML 행렬:

- `T_AD`: Vive world/raw frame을 robot base frame으로 올리는 base calibration
- `T_BC`: robot EE에서 tracker/tool frame까지의 offset
- `R_Adj`: VR point cloud와 robot point cloud의 미세 기울어짐/축 정렬 보정
- `T_FIX`: z-plane residual을 줄이기 위한 left-multiplied rigid correction
- `T_CE`: final constant offset. `T_CE[2,3]` is stored as a positive z correction knob.
- `T_SA`: orientation display/alignment용 right-multiplied rotation correction

## 기본 실행

빌드:

```bash
cd <your_ros2_workspace>
colcon build --packages-select vr_calibration
source install/setup.bash
```

캘리브레이션 실행:

```bash
ros2 run vr_calibration vr_calibration
```

현재 기본값은 다음과 같다.

```text
t_sa_mode = update
t_sa_max_delta_deg = 180.0
radj_enable = false        # 기본은 raw VR world를 그대로 쓰고 T_AD가 base-station/world frame 차이를 흡수
radj_sample_count = 0        # 0 또는 음수면 전체 captured sample 사용
capture_hold_time_s = 1.5
capture_min_hold_time_s = 0.8
capture_window_s = 0.5
capture_min_clean_samples = 20
vr_capture_age_s = 0.2
max_capture_sync_dt_s = 0.05
capture_max_vr_std_mm = 10.0
gravity_quality_select_enable = true
gravity_quality_min_pose_samples = 16
gravity_quality_max_pose_samples = 24    # 0 또는 음수면 good-quality pose 전체 사용
gravity_quality_force_residual_max_n = 3.0
gravity_quality_torque_residual_max_nm = 0.35
handeye_outlier_reject_enable = true
handeye_outlier_max_reject = 2
handeye_outlier_abs_mm = 15.0
handeye_outlier_mad_sigma = 4.0
z_fix_enable = true
z_residual_enable = true
z_residual_max_correction_mm = 10.0
max_calib_position_rms_mm = 50.0
```

`T_SA` update는 기본값으로 켜져 있으므로 별도 옵션 없이 실행하면 된다.

## 캡처 로직

노드는 waypoint 파일에서 `holding_time_s > 0`인 point만 target으로 사용한다. 각 target마다 robot이 다음 조건을 만족하면 hold 상태로 들어간다.

- position error <= `pos_enter_mm_`
- orientation error <= `ori_enter_deg_`
- robot linear velocity <= `vel_thresh_mms_`
- robot angular velocity <= `angvel_thresh_dps_`

패치 이후에는 hold가 끝나는 순간의 단일 샘플을 바로 쓰지 않는다. hold 중 다음 조건을 만족하는 clean sample만 buffer에 쌓는다.

- `/ur10skku/currentP`가 fresh
- `/raw_pose`가 `vr_capture_age_s` 이내
- `abs(currentP_time - raw_pose_time) <= max_capture_sync_dt_s`
- robot이 target region 안에 있음
- robot이 stopped 상태임

그 뒤 clean sample이 최소 `capture_min_clean_samples`개 이상이고, buffer 시간 폭이 `capture_window_s` 이상이면
buffer 안에서 가장 안정적인 `capture_window_s` 구간을 골라 평균 pose를 하나 만든다.
best window는 VR position std, robot linear/angular velocity, target dist/angle을 함께 점수화해서 선택한다.

- robot pose: clean sample 평균
- VR position: clean sample 평균
- VR orientation: quaternion sign-align 평균
- VR position std가 `capture_max_vr_std_mm`를 넘으면 캡처를 보류
- target을 떠나는 순간에도 clean buffer가 이미 유효하면 `[OUT_CAPTURE]`로 그 window를 저장하고 다음 target으로 진행한다.
- hand-eye 초벌 solve 뒤 residual이 큰 sample은 `[OUTLIER]`로 최대 `handeye_outlier_max_reject`개까지 제외하고 다시 solve한다.
- `T_FIX` 뒤에도 XY 위치별 Z 오차가 남으면 `Z_RESIDUAL` quadratic_xy 모델을 저장한다.
  runtime은 이 모델이 있으면 `T_FIX` 적용 직후 z값만 `z += f(x,y)`로 보정한다.

캡처 로그 예:

```text
[CLEAN_CAPTURE] averaged 42 samples over 0.510s | dist=0.03mm ang=0.01deg vr_std=1.25mm
[CAPTURE] target 12/32 ...
```

## Calibration 계산 순서

캡처된 sample은 내부적으로 다음 의미를 가진다.

```text
T_AB[i] = robot base(A) -> EE(B)
T_DC[i] = VR world(D) -> tracker(C)
```

전체 계산 흐름:

```text
1. clean sample set 수집
2. R_Adj 적용 여부 결정
   - 기본값 `radj_enable=false`: R_Adj=Identity
   - legacy/debug `radj_enable=true`: captured position cloud로 R_Adj 계산
3. T_DC_adj[i] = T_Adj * T_DC[i]
   where T_Adj rotation = R_Adj.T
4. hand-eye solve로 T_BC 계산
5. 각 sample에서 T_AD_i = T_AB[i] * T_BC * inv(T_DC_adj[i]) 계산
6. T_AD_i 평균으로 T_AD 생성
7. T_FIX 계산
8. runtime-chain residual 검증
9. YAML 저장
```

기본값은 `radj_enable=false`다. 이 모드에서는 base station 조합이나 새 PC의 SteamVR world frame이 달라도
raw VR world를 그대로 두고 `T_AD`가 그 차이를 직접 흡수한다. tracker-to-TCP offset이 있는 상태에서 자세가 많이
바뀌는 waypoint를 쓰면 position-only `R_Adj`가 오히려 틀어질 수 있으므로 일반 캘리브레이션은 이 기본값을 권장한다.

기존 position-cloud `R_Adj` 방식을 실험하려면 `radj_enable=true`를 켠다. 이때 `radj_sample_count=0`이면
`R_Adj` 계산에 captured sample 전체를 사용한다. 특정 개수만 쓰고 싶으면 양수로 지정한다.

```bash
ros2 run vr_calibration vr_calibration --ros-args \
  -p radj_enable:=true \
  -p radj_sample_count:=32
```

## Runtime pose 의미

`vr_calibration`은 YAML만 만든다. `/calibrated_pose`를 어떤 의미로 publish할지는 `vive_tracker_ros2/vive_tracker_node.py`의 `tool_correction_mode`가 결정한다.

```text
tool_correction_mode=none
  -> calibrated tracker/world pose publish
  -> EE와 tracker 사이 offset이 position에 남아 있음

tool_correction_mode=t_bc
  -> T_BC inverse를 적용해서 EE/TCP pose publish
  -> robot currentP와 position이 거의 같아지는 것이 정상

tool_correction_mode=t_ce
  -> legacy T_CE offset 사용
```

현재 기본값은 `none`이다. 따라서 아무 인자 없이 `vive_tracker_node`를 실행하면 `/calibrated_pose`는 robot EE pose가 아니라 tracker pose로 나온다. EE/TCP pose가 필요하면 명시적으로 `t_bc`를 켠다.

`apply_T_CE_extra=true`이면 `T_CE`가 최종 단계에서 추가 적용된다. YAML의 `T_CE[2,3]` 값을 `+dz`만큼 키우면 published z가 대략 `dz`만큼 내려간다. `vive_tracker_node`는 YAML 변경을 감지해서 실행 중에도 `T_CE`를 다시 로드한다.

```bash
ros2 run vive_tracker_ros2 vive_tracker_node --ros-args \
  -p tool_correction_mode:=t_bc
```

## 확인 포인트

캘리브레이션이 정상적으로 끝나면 다음 로그를 확인한다.

```text
[R_ADJ_DONE] multi-point position fit using N/N samples
[T_FIX] z-plane rigid fix computed ...
[CALIB_VALIDATE] runtime-chain position fit: rms=... max=...
[YAML_SAVED] ...
```

`T_SA` update를 켠 경우에는 다음 로그가 있어야 한다.

```text
[T_SA_DONE] ...
[T_SA] Pre-capture update done.
```

다음 로그가 반복되면 clean sample 조건이 너무 빡빡한 것이다.

```text
[WAIT_CLEAN_CAPTURE] clean_samples=...
[WAIT_CLEAN_CAPTURE] VR position std ... exceeds ...
```

이 경우 먼저 `/raw_pose` publish rate와 tracking 상태를 확인하고, 필요하면 `max_capture_sync_dt_s`, `capture_min_clean_samples`, `capture_max_vr_std_mm`를 완화한다.
