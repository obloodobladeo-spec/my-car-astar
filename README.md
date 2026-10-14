# TurtleBot3 A* 주행과 마우스 감지 정지

이 저장소에는 직접 실행하는 Python 노드 3개만 있습니다. `setup.py`나 colcon 패키지 파일 없이 실행할 수 있습니다. YOLO에서 말하는 `mouse`는 **컴퓨터 마우스**입니다. 동물 쥐를 검출하는 모델이 아닙니다.

## 1. 초기 설정

### 운영체제와 ROS 의존성

Ubuntu 22.04와 ROS 2 Humble 기준입니다. ROS 2 apt 저장소가 이미 등록된 환경에서 필요한 Python/ROS 패키지를 설치합니다. 이미 설치되어 있다면 다시 설치할 필요가 없습니다.

```bash
sudo apt update
sudo apt install ros-humble-ros-base python3-numpy \
  ros-humble-rclpy ros-humble-geometry-msgs ros-humble-nav-msgs \
  ros-humble-sensor-msgs ros-humble-std-msgs ros-humble-tf2-ros
```

Gazebo와 Nav2 시뮬레이션도 새로 준비해야 하는 PC라면 다음 패키지가 필요합니다.

```bash
sudo apt install ros-humble-turtlebot3-gazebo \
  ros-humble-turtlebot3-description ros-humble-nav2-bringup
```

이 노드들은 `/map`, `/amcl_pose`, `map → base_link` TF, `/scan`, `/camera/image_raw`가 이미 발행되는 환경을 사용합니다. 맵 서버와 AMCL이 없으면 A* 경로를 만들 수 없습니다. 로봇이 움직이려면 `/cmd_vel`을 받는 TurtleBot3 또는 Gazebo 모델도 필요합니다.

### 코드와 YOLO 준비

```bash
git clone https://github.com/obloodobladeo-spec/my-car-astar.git ~/my-car-astar
```

마우스 감지 노드에는 Ultralytics와 OpenCV가 필요합니다. **기존 `~/yolo_webcam/.venv`를 사용하며 새 가상환경을 만들 필요는 없습니다.** 해당 환경에 아직 설치되지 않았다면 다음 명령을 실행합니다.

```bash
~/yolo_webcam/.venv/bin/python -m pip install ultralytics opencv-python
```

모델 파일은 저장소에 포함하지 않습니다. 현재 PC에는 `~/yolo_webcam/yolov8n.pt`가 있습니다. 다른 PC에서 파일이 없다면 아래 명령으로 받을 수 있습니다.

```bash
mkdir -p ~/yolo_webcam
cd ~/yolo_webcam
~/yolo_webcam/.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

감지 노드는 `cv_bridge`를 사용하지 않고 `sensor_msgs/Image`를 NumPy/OpenCV 영상으로 직접 변환합니다.

## 2. 실행

모든 터미널에서 같은 `ROS_DOMAIN_ID`를 사용하세요. 아래 `turtlebot3_ws`와 `ros2_ws`의 `source` 명령은 **현재 PC에 해당 워크스페이스가 있을 때** 실행합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=12
export TURTLEBOT3_MODEL=waffle_pi
```

**터미널 1 — Gazebo, 맵 서버, AMCL, RViz**

현재 PC의 기존 런치 파일을 사용할 때:

```bash
ros2 launch my_turtle_pkg my_turtle_navi.launch.py
```

이 런치 파일은 이 저장소에 포함되지 않습니다. 다른 PC에서는 자신의 Gazebo·맵 서버·AMCL 런치를 사용하세요. RViz의 **2D Pose Estimate**로 맵 위의 로봇 초기 위치와 방향을 먼저 설정합니다.

**터미널 2 — 카메라 마우스 감지**

```bash
cd ~/my-car-astar
~/yolo_webcam/.venv/bin/python mouse_detector_node.py \
  --ros-args -p use_sim_time:=true
```

모델 위치가 다르다면 `-p model_path:=/절대/경로/yolov8n.pt`를 덧붙입니다. 카메라 토픽이 다르다면 `-p image_topic:=/카메라/토픽`을 덧붙입니다.

**터미널 3 — A* 주행 및 정지 제어**

```bash
cd ~/my-car-astar
python3 best_driver_node.py --ros-args -p use_sim_time:=true
```

**터미널 4 — 목표 전송**

```bash
ros2 topic pub --once /astar_goal geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: -4.0536, y: 4.6214}, orientation: {w: 1.0}}}'
```

다음 명령으로 연결 상태를 확인할 수 있습니다.

```bash
ros2 topic info /map
ros2 topic info /camera/image_raw
ros2 topic echo /mouse_detected
ros2 topic echo /cmd_vel
```

`/cmd_vel`을 발행하는 다른 자율주행 노드를 동시에 실행하지 마세요.

## 3. 노드 설명

| 파일 | 역할 |
| --- | --- |
| `astar_pursuit.py` | `/map`을 2차원 격자로 바꾸고 장애물 여유 반경을 적용합니다. A*로 `/astar_goal`까지의 경로를 만든 뒤 `/astar_path`로 발행하고, 경로를 따라 `/cmd_vel`을 계산합니다. `best_driver_node.py`가 이 코드를 가져다 씁니다. |
| `best_driver_node.py` | A* 주행에 `/scan` 장애물 회피와 `/mouse_detected` 정지를 추가합니다. 마우스가 보이면 즉시 정지하고, 사라졌다는 메시지가 오면 경로 주행을 재개합니다. |
| `mouse_detector_node.py` | 최신 `/camera/image_raw` 프레임만 처리합니다. `yolov8n.pt`의 COCO `mouse` 클래스(64)만 결과로 받아 `/mouse_detected`를 발행합니다. |

### 주행과 장애물 회피

주행 노드는 0.1초마다 현재 위치와 경로의 다음 지점을 비교합니다. 방향 차이가 크면 제자리 회전하고, 작으면 전진하면서 방향을 보정합니다. 전방 장애물이 기본 `0.32m`보다 가까우면 넓은 쪽으로 `±0.7rad/s` 제자리 회전을 합니다. 전방이 `0.45m` 이상인 스캔을 3번 연속 받으면, 감지한 장애물 지점을 임시 맵에 표시하고 현재 위치에서 A* 경로를 다시 계산합니다.

마우스를 한 번 감지하면 바로 정지합니다. 감지 노드에서 마우스가 3프레임 연속 보이지 않을 때 `/mouse_detected=false`가 발행되어 주행이 재개됩니다. 감지 노드가 중단되어 `false`가 오지 않으면 정지 상태가 유지됩니다. YOLO의 `classes=[64]`는 출력 클래스를 거르는 옵션이므로 전체 모델 추론량 자체를 줄이지는 않습니다.

### 주요 파라미터

`best_driver_node.py` 실행 명령 뒤에 `--ros-args -p 이름:=값`으로 지정할 수 있습니다.

| 파라미터 | 기본값 | 의미 |
| --- | ---: | --- |
| `linear_velocity` | `0.08` | 최대 전진 속도 (m/s) |
| `lookahead_distance` | `0.35` | 경로에서 앞쪽 목표점을 고르는 거리 (m) |
| `robot_clearance` | `0.24` | 맵 장애물에 적용하는 여유 반경 (m) |
| `goal_tolerance` | `0.15` | 목표 도착 허용 거리 (m) |
| `front_stop_distance` | `0.32` | 라이다 회피 시작 거리 (m) |
| `front_resume_distance` | `0.45` | 회피 종료 거리 (m) |
| `avoid_turn_speed` | `0.7` | 회피 중 제자리 회전 속도 (rad/s) |
| `scan_timeout` | `1.0` | 오래된 라이다 데이터를 거부하는 시간 (s) |

카메라 노드는 `model_path`, `image_topic`, `confidence`(기본 `0.5`), `process_period`(기본 `0.2`초)를 받습니다.

## 4. 현재 한계

- A*가 사용하는 맵과 실제 Gazebo/로봇 위치가 맞아야 합니다. 시작 셀이나 목표 셀이 장애물 여유 영역 안에 있으면 `No route`가 나옵니다.
- 라이다로 임시 추가한 장애물 영역이 로봇의 현재 셀까지 덮으면 안전 정지 상태에서 기다릴 수 있습니다. 이 복구 분기는 아직 개선되지 않았습니다.
- `mouse`는 컴퓨터 마우스 클래스입니다. 인식 결과가 잘못될 수 있으므로 이 기능을 사람이나 실제 장애물에 대한 유일한 안전장치로 사용하지 마세요.
