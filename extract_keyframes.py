import cv2
import os
import argparse
import yt_dlp
import io
import img2pdf
import ocrmypdf
from PIL import Image
from scenedetect import open_video, SceneManager, ContentDetector

THRESHOLD = 21.0
MIN_SCENE_LEN = 10


def download_youtube_video(url, output_dir):
    print(f"Обнаружена ссылка на YouTube. Начинаю скачивание: {url}")
    os.makedirs(output_dir, exist_ok=True)

    # Настройки: скачиваем только видео (без аудио) в формате mp4 для скорости
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best',
        'format_sort': ['vcodec:h264'],  # Жесткий приоритет кодеку H.264 (avc1)
        'outtmpl': os.path.join(output_dir, '%(title)s_%(id)s.%(ext)s'),
        'overwrites': True,  # ВАЖНО: принудительно перезаписать старый сломанный файл
        'quiet': False,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info_dict)

    print(f"Видео успешно скачано: {video_path}")
    return video_path


def extract_scene_frames(video_path, output_dir, num_frames):
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл '{video_path}' не найден.")
        return

    print(f"Анализ видео '{video_path}'...")

    video = open_video(video_path)
    scene_manager = SceneManager()

    # 1. ОТКЛЮЧАЕМ АВТОМАТИКУ
    scene_manager.auto_downscale = False
    # 2. Ускорение: Включаем сильное уменьшение картинки перед анализом
    scene_manager.downscale = 8  # Работает в десятки раз быстрее
    # 3. Оптимизация детектора:
    scene_manager.add_detector(ContentDetector(threshold=THRESHOLD, min_scene_len=MIN_SCENE_LEN))
    # 4. Ускорение: Заставляем SceneManager пропускать кадры при чтении
    scene_manager.detect_scenes(video, frame_skip=2)

    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        print("Сцены не найдены. Возможно, видео слишком короткое или статичное.")
        return

    print(f"Найдено сцен: {len(scene_list)}. Начинаю извлечение кадров...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(output_dir, exist_ok=True)

    extracted_count = 0
    frames_in_memory = []

    for i, (start_time, end_time) in enumerate(scene_list):
        scene_num = i + 1

        start_frame = start_time.frame_num
        end_frame = end_time.frame_num
        length = end_frame - start_frame

        frames_to_extract = []

        # Нам нужно отступить на 3 кадра от end_frame, чтобы избежать
        # возможного черного экрана (fade out) перед самой склейкой
        last_frame_of_scene = max(start_frame, end_frame - 3)

        if num_frames == 1:
            # Если нужен только 1 кадр — берем середину
            frames_to_extract.append(start_frame + length // 2)

        elif num_frames == 2:
            # Если 2 кадра — берем строго начало и самый конец сцены
            frames_to_extract.append(start_frame)
            frames_to_extract.append(last_frame_of_scene)

        else:
            # Если 3 и более кадров — берем начало, конец и равномерно между ними
            frames_to_extract.append(start_frame)

            # Считаем шаг для промежуточных кадров
            # Делим на (num_frames - 1), чтобы последний кадр точно попал в конец отрезка
            step = length / (num_frames - 1)

            for j in range(1, num_frames - 1):
                # Округляем до целого числа кадров
                middle_idx = int(start_frame + j * step)
                frames_to_extract.append(middle_idx)

            frames_to_extract.append(last_frame_of_scene)

        # Удаляем возможные дубликаты индексов (если сцена очень короткая)
        # и сортируем по порядку воспроизведения
        frames_to_extract = sorted(list(set(frames_to_extract)))

        # Обработка кадров в оперативной памяти
        for j, frame_idx in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # 1. Накладываем таймкод
                time_in_seconds = frame_idx / fps
                mins, secs = divmod(time_in_seconds, 60)
                hours, mins = divmod(mins, 60)
                time_str = f"Scene_{scene_num} {int(hours):01d}:{int(mins):02d}:{secs:05.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, time_str, (30, 60), font, 1.5, (250, 250, 200), 70)
                cv2.putText(frame, time_str, (30, 60), font, 1.5, (10, 10, 10), 3)

                # 2. Кодируем кадр в JPEG прямо в оперативной памяти
                # cv2.imencode возвращает кортеж: (статус, массив байтов)
                ret_encode, buffer = cv2.imencode('.jpg', frame)
                if ret_encode:
                    # buffer.tobytes() превращает массив numpy в сырые байты картинки
                    frames_in_memory.append(buffer.tobytes())

                extracted_count += 1

    cap.release()
    print(f"Успешно извлечено {extracted_count} кадров в оперативную память.")

    # --- СБОРКА PDF И РАСПОЗНАВАНИЕ ТЕКСТА ---
    if frames_in_memory:
        # Извлекаем чистое имя файла из пути (например, "video_name.mp4" -> "video_name")
        file_name_with_ext = os.path.basename(video_path)
        base_name, _ = os.path.splitext(file_name_with_ext)

        final_pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

        print(f"Сборка чернового PDF из {len(frames_in_memory)} кадров в памяти...")

        # 1. Собираем PDF в оперативной памяти (передаем напрямую список байтов от cv2)
        raw_pdf_bytes = img2pdf.convert(frames_in_memory)

        # 2. Передаем виртуальный файл в OCRmyPDF
        print(f"Запускаю OCR (ocrmypdf)... Это займет некоторое время.")

        # Оборачиваем сырые байты в виртуальный файл (BytesIO), который ocrmypdf сможет прочитать
        pdf_input_stream = io.BytesIO(raw_pdf_bytes)

        try:
            # Вызываем OCRmyPDF
            ocrmypdf.ocr(
                pdf_input_stream,
                final_pdf_path,
                language='eng',
                force_ocr=True,
                progress_bar=False
            )
            print(f"Готово! Финальный PDF с текстом сохранен как: {final_pdf_path}")
        except Exception as e:
            print(f"Ошибка при создании OCR PDF: {e}")

            # Если OCR сломался, сохраняем черновой PDF на диск
            raw_pdf_path = os.path.join(output_dir, f"{base_name}_raw.pdf")
            with open(raw_pdf_path, "wb") as f:
                f.write(raw_pdf_bytes)
            print(f"Черновой PDF без поиска сохранен как: {raw_pdf_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скачивание с YouTube и извлечение ключевых кадров в GIF.")
    parser.add_argument("source", help="Путь к локальному видеофайлу ИЛИ ссылка на YouTube")
    parser.add_argument("-o", "--output", default="frames", help="Папка для сохранения (по умолчанию: 'frames')")
    parser.add_argument("-n", "--num-frames", type=int, default=2, help="Количество кадров на сцену (по умолчанию: 2)")

    args = parser.parse_args()
    input_source = args.source

    # Проверяем, является ли источник ссылкой
    if input_source.startswith(('http://', 'https://')):
        try:
            # Скачиваем видео и получаем путь к скачанному файлу
            video_path = download_youtube_video(input_source, args.output)
        except Exception as e:
            print(f"Произошла ошибка при скачивании: {e}")
            exit(1)
    else:
        video_path = input_source

    # Запускаем извлечение кадров и сборку
    extract_scene_frames(video_path, args.output, args.num_frames)
