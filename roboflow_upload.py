import os
from roboflow import Roboflow
import cv2
import supervision as sv

# ==============================================================================
# [1. 설정 구간]
# ==============================================================================

API_KEY = "HJMrtyzyTtA6TMKpMEHV"
WORKSPACE_ID = "minsun"
PROJECT_ID = "metric-szakz"

# 업로드할 영상 파일 하나만 지정
VIDEO_PATH = r"C:\Users\nva_kist\Desktop\minsun\KIST\Tomato_detect\testset\0407.mp4"

# 몇 fps로 샘플링할지
TARGET_FPS = 5

# ==============================================================================

def main():
    print("🔄 Roboflow에 연결 중...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 영상 파일이 없습니다: {VIDEO_PATH}")
        return

    video_name_pure = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    unique_video_name = video_name_pure

    print(f"\n🎥 영상 업로드 시작: {unique_video_name}")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

    # 원본 fps 기준으로 5fps 샘플링
    frame_interval = max(1, int(round(video_info.fps / TARGET_FPS)))

    print(f"원본 FPS: {video_info.fps}")
    print(f"샘플링 FPS: {TARGET_FPS}")
    print(f"FRAME_INTERVAL: {frame_interval}")

    frame_count = 0
    upload_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                temp_filename = f"{unique_video_name}_frame_{frame_count:04d}.jpg"

                cv2.imwrite(temp_filename, frame)

                if os.path.exists(temp_filename):
                    project.upload(
                        image_path=temp_filename,
                        split="train",
                        batch_name=unique_video_name
                    )

                    os.remove(temp_filename)
                    upload_count += 1
                    print(".", end="", flush=True)

            frame_count += 1

        print(f"\n✅ [{unique_video_name}] 업로드 완료: {upload_count}장")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")

    finally:
        cap.release()

if __name__ == "__main__":
    main()