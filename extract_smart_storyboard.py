import cv2
import os
import argparse
import yt_dlp
import img2pdf
import pytesseract
import ocrmypdf
import re
from langdetect import detect_langs, DetectorFactory
from scenedetect import open_video, SceneManager, ContentDetector
from rapidfuzz import process, fuzz

THRESHOLD = 46.0
MIN_SCENE_LEN = 15
SIMILARITY_SCORE = 50.0
# Фиксируем seed для langdetect, чтобы результаты были воспроизводимы
DetectorFactory.seed = 0

# Словарь для перевода ISO-кодов langdetect (2 буквы) в коды Tesseract (3 буквы)
# Добавьте сюда другие языки, если ожидаете их в видео (испанский, итальянский и т.д.)
LANG_MAP = {
    'ru': 'rus',
    'en': 'eng'
    # 'de': 'deu',
    # 'fr': 'fra',
    # 'uk': 'ukr',
    # 'es': 'spa',
    # 'it': 'ita'
}


def download_youtube_video(url, output_dir):
    print(f"Скачивание YouTube видео: {url}")
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best',
        'format_sort': ['vcodec:h264'],  # Принудительно H.264 для OpenCV
        'outtmpl': os.path.join(output_dir, '%(title)s_%(id)s.%(ext)s'),
        'overwrites': True,
        'quiet': False,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info_dict)

    print(f"Видео скачано: {video_path}")
    return video_path


def is_valid_text(text):
    """
    Проверяет, является ли текст осмысленным (отсеивает мусор вроде ', . | \').
    Требует минимум 5 букв или цифр.
    """
    # Удаляем все не-буквы и не-цифры
    alphanumeric = re.sub(r'[^a-zA-Zа-яА-Я0-9]', '', text)
    return len(alphanumeric) >= 5


def detect_tesseract_langs(text):
    """
    Определяет языки в тексте и возвращает их коды в формате Tesseract.
    """
    detected_tess_langs = set()
    try:
        # Получаем список языков с вероятностями
        langs = detect_langs(text)

        for lang_obj in langs:
            # Берем язык только если уверенность нейросети > 20%
            if lang_obj.prob > 0.2:
                iso_lang = lang_obj.lang
                if iso_lang in LANG_MAP:
                    detected_tess_langs.add(LANG_MAP[iso_lang])
                else:
                    # Если язык не в нашем словаре, по умолчанию добавляем английский
                    detected_tess_langs.add('eng')
    except Exception:
        # Если langdetect упал (например, текст состоит только из цифр),
        # возвращаем базовые языки
        detected_tess_langs.update(['rus', 'eng'])

    return detected_tess_langs


def process_video_to_smart_pdf(video_path, output_dir, num_frames=2):
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл '{video_path}' не найден.")
        return

    print(f"Анализ сцен в видео '{video_path}'...")

    video = open_video(video_path)
    scene_manager = SceneManager()

    # 1. ОТКЛЮЧАЕМ АВТОМАТИКУ
    scene_manager.auto_downscale = False

    # 2. Ускорение: Включаем сильное уменьшение картинки перед анализом
    scene_manager.downscale = 8  # Работает в десятки раз быстрее

    # 3. Оптимизация детектора:
    # min_scene_len=15 защищает от ложных склеек при моргании света
    scene_manager.add_detector(ContentDetector(threshold=THRESHOLD, min_scene_len=MIN_SCENE_LEN))

    # 4. Ускорение: Заставляем SceneManager пропускать кадры при чтении
    # frame_skip=2 означает, что мы анализируем только каждый 3-й кадр (ускорение в 3 раза)
    scene_manager.detect_scenes(video, frame_skip=2)

    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        print("Сцены не найдены.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(output_dir, exist_ok=True)

    frames_in_memory = []
    global_detected_langs = set()  # Собираем все уникальные языки из всего видео
    saved_texts_archive = []  # Массив всех уникальных текстов, которые уже пошли в PDF

    print(f"Найдено сцен: {len(scene_list)}. Начинаю умное извлечение кадров...")

    for i, (start_time, end_time) in enumerate(scene_list):
        scene_num = i + 1

        start_frame = start_time.frame_num
        end_frame = end_time.frame_num
        length = end_frame - start_frame

        frames_to_extract = []
        # Нам нужно отступить на 1 кадр от end_frame, так как end_frame
        # указывает на самый первый кадр СЛЕДУЮЩЕЙ сцены
        last_frame_of_scene = max(start_frame, end_frame - 1)

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

        for j, frame_idx in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # 1. Черновое OCR для проверки наличия текста
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_text = pytesseract.image_to_string(rgb_frame, lang='rus+eng').strip()

            # 2. Фильтрация: если текста нет или это мусор — пропускаем кадр
            if not is_valid_text(raw_text):
                print(f"Сцена {scene_num} | Кадр {j + 1}: Текст не найден. Кадр отброшен.")
                continue

            print(f"Сцена {scene_num} | Кадр {j + 1}: Найден текст, сохраняю.")

            # --- БЛОК АНТИ-ДУБЛИКАТА (ПОИСК ПО АРХИВУ) ---
            normalized_text = " ".join(raw_text.split())

            if saved_texts_archive:
                # process.extractOne мгновенно находит самую похожую строку в массиве
                # Используем fuzz.ratio (простое посимвольное совпадение)
                # RapidFuzz возвращает кортеж: (найденная_строка, процент_сходства_от_0_до_100, индекс)
                best_match = process.extractOne(normalized_text, saved_texts_archive, scorer=fuzz.ratio)

                if best_match:
                    similarity_score = best_match[1]
                    # Если текст похож на любой из предыдущих на 85% и более
                    if similarity_score >= SIMILARITY_SCORE:
                        print(
                            f"Сцена {scene_num} | Кадр {j + 1}: Текст уже был ранее (сходство {similarity_score:.1f}%). Дубликат отброшен.")
                        continue

            # Если дошли сюда, текст действительно уникален для всего видео
            saved_texts_archive.append(normalized_text)
            print(f"Сцена {scene_num} | Кадр {j + 1}: Найден новый уникальный текст, сохраняю.")
            # ----------------------------------------------

            # 3. Детектируем реальные языки на этом кадре
            frame_langs = detect_tesseract_langs(raw_text)
            global_detected_langs.update(frame_langs)

            # 4. Накладываем таймкод
            time_in_seconds = frame_idx / fps
            mins, secs = divmod(time_in_seconds, 60)
            hours, mins = divmod(mins, 60)
            time_str = f"Scene_{scene_num} {int(hours):01d}:{int(mins):02d}:{secs:05.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, time_str, (30, 60), font, 1.5, (250, 250, 200), 70)
            cv2.putText(frame, time_str, (30, 60), font, 1.5, (10, 10, 10), 3)

            # 5. Кодируем кадр в JPEG в оперативной памяти
            ret_encode, buffer = cv2.imencode('.jpg', frame)
            if ret_encode:
                frames_in_memory.append(buffer.tobytes())

    cap.release()

    if not frames_in_memory:
        print("В видео не найдено ни одного кадра с осмысленным текстом. PDF не создан.")
        return

    # --- СБОРКА И OCR ---

    # Всегда добавляем английский как базовый запасной вариант
    global_detected_langs.add('eng')

    # Формируем строку языков для OCR (например, "rus+eng+deu")
    ocr_lang_str = "+".join(global_detected_langs)
    print(f"\nИзвлечено полезных кадров: {len(frames_in_memory)}.")
    print(f"Итоговые языки для OCR-обработки: {ocr_lang_str}")

    # Извлекаем чистое имя файла из пути (например, "video_name.mp4" -> "video_name")
    file_name_with_ext = os.path.basename(video_path)
    base_name, _ = os.path.splitext(file_name_with_ext)
    raw_pdf_path = os.path.join(output_dir, f"{base_name}_raw.pdf")
    final_pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    # Сначала сохраняем обычный PDF из картинок
    with open(raw_pdf_path, "wb") as f:
        f.write(img2pdf.convert(frames_in_memory))

    print(f"Черновой PDF сохранен. Запускаю OCR (ocrmypdf)...")

    try:
        # Накладываем текстовый слой (делаем searchable PDF)
        ocrmypdf.ocr(
            raw_pdf_path,
            final_pdf_path,
            language=ocr_lang_str,
            force_ocr=True,
            progress_bar=False
        )
        print(f"Готово! Финальный PDF с текстом сохранен как: {final_pdf_path}")

        # Удаляем черновой PDF, чтобы не засорять диск
        os.remove(raw_pdf_path)
    except Exception as e:
        print(f"Ошибка при создании OCR PDF: {e}")
        print(f"Черновой PDF без поиска оставлен по пути: {raw_pdf_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Умная раскадровка видео с OCR и фильтрацией пустых кадров.")
    parser.add_argument("source", help="Путь к локальному видеофайлу ИЛИ ссылка на YouTube")
    parser.add_argument("-o", "--output", default="storyboards",
                        help="Папка для сохранения (по умолчанию: 'storyboards')")
    parser.add_argument("-n", "--num-frames", type=int, default=2,
                        help="Кадров на сцену для проверки (по умолчанию: 2)")

    args = parser.parse_args()

    input_source = args.source

    if input_source.startswith(('http://', 'https://')):
        try:
            video_path = download_youtube_video(input_source, args.output)
        except Exception as e:
            print(f"Произошла ошибка при скачивании: {e}")
            exit(1)
    else:
        video_path = input_source

    process_video_to_smart_pdf(video_path, args.output, args.num_frames)