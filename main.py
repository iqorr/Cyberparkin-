import cv2
import numpy as np

def detect_green_entry(image):
    # Konwersja do przestrzeni HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Zakres zielonego koloru (dostosuj, jeśli potrzeba)
    lower_green = np.array([40, 50, 50])  # Dolny próg (H, S, V)
    upper_green = np.array([80, 255, 255])  # Górny próg (H, S, V)

    # Maskowanie zielonych obszarów
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Znajdowanie konturów zielonych obszarów
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filtracja na podstawie rozmiaru
        if area > 1000:  # Rozmiar minimalnego zielonego kwadratu
            x, y, w, h = cv2.boundingRect(contour)

            # Rysowanie prostokąta wokół zielonego kwadratu
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Wjazd", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def check_parking_occupancy(image, parking_spaces):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    occupied_spaces = 0

    for i, (x, y, w, h) in enumerate(parking_spaces):
        roi = gray[y:y + h, x:x + w]

        # Analiza średniej jasności i krawędzi
        mean_brightness = cv2.mean(roi)[0]
        edges = cv2.Canny(roi, 50, 150)
        edge_count = cv2.countNonZero(edges)
        total_pixels = w * h
        edge_ratio = edge_count / total_pixels

        # Decyzja o zajętości
        if  edge_ratio > 0.05:
            color = (0, 0, 255)  # Zajęte miejsce (czerwone)
            occupied_spaces += 1
        else:
            color = (0, 255, 0)  # Puste miejsce (zielone)

        # Rysowanie prostokątów i numerów
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if (i+1) % 2 == 1 or (i+1) == 6:
            text_x = x - 30
        else:
            text_x = x + w + 10

        text_y = y + h // 2
        cv2.putText(image, f"{i + 1}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)  # Czarny kontur
        cv2.putText(image, f"{i + 1}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)  # Właściwy kolor

    print(f"Zajętych miejsc: {occupied_spaces} / {len(parking_spaces)}")
    return image


def split_image(image):
    height, width, _ = image.shape

    # Przycięcie czarnych boków
    crop_x_start = 420
    crop_x_end = width - 420
    cropped_image = image[:, crop_x_start:crop_x_end]

    # Podział obrazu na części
    height, width, _ = cropped_image.shape
    left_part = cropped_image[:, :width // 3]
    left_top = left_part[:height // 2, :]  # Wyjazd
    left_bottom = left_part[height // 2:, :]  # Wjazd
    parking_area = cropped_image[:, (width // 3) - 45:]  # Miejsca parkingowe

    return left_top, left_bottom, parking_area

def detect_cars(image, parking_spaces, background_frame):
    # Obliczanie różnicy między bieżącą klatką a obrazem referencyjnym
    gray_current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_background, gray_current)

    # Progowanie różnic
    _, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Znajdowanie konturów różnic
    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)


        if area > 500 and w < 350 and h < 350 and w > 50 and h > 50:

            # Ignorowanie obszarów pokrywających się z miejscami parkingowymi
            car_overlaps_parking_space = False
            for (px, py, pw, ph) in parking_spaces:
                if x < px + pw and x + w > px and y < py + ph and y + h > py:
                    car_overlaps_parking_space = True
                    break

            # Rysowanie żółtego prostokąta dla wykrytych samochodów
            if not car_overlaps_parking_space:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Żółty prostokąt
                cv2.putText(image, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return image


def detect_and_number_parking_spaces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    parking_spaces = []
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 25000 and area < 50000:  # Obniżony próg

            x, y, w, h = cv2.boundingRect(contour)

            # Proporcje prostokąta
            aspect_ratio = w / h
            if 1.0 < aspect_ratio < 2.5:
                roi = gray[y:y + h, x:x + w]
                edges = cv2.Canny(roi, 50, 150)
                edge_count = cv2.countNonZero(edges)
                total_pixels = w * h
                edge_ratio = edge_count / total_pixels

                if edge_ratio < 0.5:  # Maksymalna ilość krawędzi w pustym miejscu
                    parking_spaces.append((x, y, w, h))

    print(f"Znaleziono {len(parking_spaces)} miejsc parkingowych.")
    return parking_spaces


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo!")
        return

    # Pobranie pierwszej klatki jako obraz referencyjny
    ret, first_frame = cap.read()
    if not ret:
        print("Nie można wczytać pierwszej klatki!")
        return

    _, _, reference_parking_area = split_image(first_frame)
    parking_spaces = detect_and_number_parking_spaces(reference_parking_area)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Kopia pełnego obrazu do nanoszenia wyników
        full_frame_with_annotations = frame.copy()

        # Podział obrazu na części
        left_top, left_bottom, parking_area = split_image(frame)

        # Wykrywanie wjazdu na parking (zielony kwadrat)
        full_frame_with_annotations = detect_green_entry(full_frame_with_annotations)

        # Sprawdzanie zajętości miejsc parkingowych
        annotated_parking_area = check_parking_occupancy(parking_area.copy(), parking_spaces)

        # Wykrywanie samochodów w części parkingowej
        annotated_parking_area_with_cars = detect_cars(annotated_parking_area.copy(), parking_spaces, reference_parking_area)

        # Dopasowanie rozmiaru części parkingowej do obszaru w pełnym obrazie
        parking_start_x = frame.shape[1] - parking_area.shape[1] - 420  # Pozycja parkingu w pełnym obrazie
        target_width = frame.shape[1] - parking_start_x
        if annotated_parking_area_with_cars.shape[1] != target_width:
            annotated_parking_area_with_cars = cv2.resize(annotated_parking_area_with_cars,
                                                          (target_width, parking_area.shape[0]),
                                                          interpolation=cv2.INTER_LINEAR)

        # Nanoszenie wyników na pełny obraz
        full_frame_with_annotations[:, parking_start_x:] = annotated_parking_area_with_cars

        # Wyświetlanie wyników na pełnej klatce
        scale_percent = 60  # Skalowanie obrazu
        width = int(full_frame_with_annotations.shape[1] * scale_percent / 100)
        height = int(full_frame_with_annotations.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_frame = cv2.resize(full_frame_with_annotations, dim, interpolation=cv2.INTER_AREA)
        resized_frame = resized_frame[::, 260:]

        cv2.imshow("Pełny obraz z oznaczeniami", resized_frame)

        # Przerwanie po naciśnięciu 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    video_path = "filmik2.mp4"
    process_video(video_path)

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def rotate_image(image, angle):
    # Funkcja do obracania obrazu o podany kąt
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Macierz obrotu
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated


def rotate_90_degrees(image):
    # Funkcja do obrotu o 90 stopni w prawo
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def read_plate_from_image(image_path):
    # Wczytanie obrazu
    image = cv2.imread(image_path)
    if image is None:
        print("Nie można wczytać obrazu!")
        return

    # Konwersja do skali szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wyświetlenie oryginalnego obrazu
    cv2.imshow("Oryginalny obraz", image)

    for angle in [-15, -10, -5, 0, 5, 10, 15, 90]:  # Obrót obrazu dla lepszego OCR
        rotated_image = rotate_image(gray, angle)

        # Pokaż obrócone obrazy dla każdego kąta obrotu
        cv2.imshow(f"Obrócony obraz (kąt: {angle})", rotated_image)

        # OCR na obróconym obrazie
        text = pytesseract.image_to_string(rotated_image, config='--psm 8')

        if text.strip():  # Jeśli OCR zwrócił tekst, wyświetl go
            print(f"Odczytana rejestracja: {text.strip()} (kąt obrotu: {angle})")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

    print("Nie znaleziono tablicy rejestracyjnej.")
    cv2.waitKey(0)  # Czekaj na klucz przed zamknięciem
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "xd222.jpg"
    read_plate_from_image(image_path)
