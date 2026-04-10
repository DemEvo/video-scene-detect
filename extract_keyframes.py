import cv2
import os
import argparse
import yt_dlp
from scenedetect import open_video, SceneManager, ContentDetector
import img2pdf
import glob
import ocrmypdf

def create_pdf_from_frames(output_dir, pdf_path):
    # Ищем все .jpg файлы в папке и сортируем их по имени (scene_001_01.jpg и т.д.)
    search_pattern = os.path.join(output_dir, '*.jpg')
    image_files = sorted(glob.glob(search_pattern))

    if not image_files:
        print(f"В папке '{output_dir}' не найдено изображений для создания PDF.")
        return

    print(f"Сборка PDF из {len(image_files)} кадров...")

    # Конвертируем список картинок в PDF
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_files))

    print(f"Готово! PDF сохранен как: {pdf_path}")


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
    scene_manager.add_detector(ContentDetector(threshold=41.0))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        print("Сцены не найдены. Возможно, видео слишком короткое или статичное.")
        return

    print(f"Найдено сцен: {len(scene_list)}. Начинаю извлечение кадров...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(output_dir, exist_ok=True)

    extracted_count = 0

    for i, (start_time, end_time) in enumerate(scene_list):
        scene_num = i + 1

        start_frame = start_time.frame_num
        end_frame = end_time.frame_num
        length = end_frame - start_frame

        frames_to_extract = []

        if num_frames == 1:
            frames_to_extract.append(start_frame + length // 2)
        else:
            step = length // num_frames
            for j in range(num_frames):
                frame_idx = start_frame + j * step
                frame_idx = min(frame_idx, end_frame - 1)
                frames_to_extract.append(frame_idx)

                # Сохранение кадров с наложением таймкода
                for j, frame_idx in enumerate(frames_to_extract):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # 1. Вычисляем таймкод (часы, минуты, секунды)
                        time_in_seconds = frame_idx / fps
                        mins, secs = divmod(time_in_seconds, 60)
                        hours, mins = divmod(mins, 60)

                        # Формируем строку: Сцена N | HH:MM:SS.мс
                        time_str = f"Scene {scene_num} | {int(hours):02d}:{int(mins):02d}:{secs:05.2f}"

                        # 2. Настройки шрифта
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        position = (30, 60)  # Отступ сверху и слева
                        font_scale = 0.8  # Размер шрифта (увеличьте, если видео 4K)
                        thickness = 2

                        # 3. Рисуем черную обводку (чтобы текст читался на светлом фоне)
                        cv2.putText(frame, time_str, position, font, font_scale, (0, 0, 0), thickness + 3)
                        # 4. Рисуем сам белый текст поверх обводки
                        cv2.putText(frame, time_str, position, font, font_scale, (255, 255, 255), thickness)

                        # 5. Сохраняем картинку
                        filename = os.path.join(output_dir, f'scene_{scene_num:03d}_{j + 1:02d}.jpg')
                        cv2.imwrite(filename, frame)
                        extracted_count += 1

    cap.release()
    print(f"Успешно завершено! Сохранено {extracted_count} изображений в папку '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скачивание с YouTube и извлечение ключевых кадров.")
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

    # Запускаем извлечение кадров
    extract_scene_frames(video_path, args.output, args.num_frames)

    # Формируем имя для PDF-файла (например, frames/storyboard.pdf)
    pdf_filename = os.path.join(args.output, "storyboard.pdf")

    # Запускаем сборку PDF
    create_pdf_from_frames(args.output, pdf_filename)

    # Создаем финальный файл с текстовым слоем
    searchable_pdf = os.path.join(args.output, "storyboard_searchable.pdf")
    print("Запуск распознавания текста (OCR)... Это может занять некоторое время.")

    try:
        # force_ocr=True заставляет движок распознавать текст даже на картинках
        ocrmypdf.ocr(pdf_filename, searchable_pdf, language="rus+eng", force_ocr=True)
        print(f"Готово! PDF с возможностью поиска сохранен как: {searchable_pdf}")
    except Exception as e:
        print(f"Ошибка при создании OCR PDF: {e}")