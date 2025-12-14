import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
from collections import deque
import time

# MediaPipe yüz tespiti için
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# EAR (Eye Aspect Ratio) hesaplama için kritik noktalar (MediaPipe Face Mesh)
# Basitleştirilmiş EAR için kullanılacak noktalar
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Kafa pozisyonu için referans noktalar
FACE_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],           # Burun ucu
    [-225.0, 170.0, -135.0],   # Sol göz
    [225.0, 170.0, -135.0],    # Sağ göz
    [-150.0, -150.0, -125.0],  # Sol ağız köşesi
    [150.0, -150.0, -125.0]    # Sağ ağız köşesi
], dtype=np.float64)

class AttentionTracker:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Dikkat metrikleri için geçmiş veriler
        self.eye_history = deque(maxlen=30)  # Son 30 frame
        self.head_pose_history = deque(maxlen=30)
        self.attention_score = 1.0  # 0.0 (dağınık) - 1.0 (odaklı)
        self.attention_history = deque(maxlen=10)  # Smoothing için
        
        # Kalibrasyon için
        self.calibrated = False
        self.base_eye_ratio = 0.25
        self.base_head_pose = np.array([0, 0, 0])
        
    def calculate_ear(self, landmarks, top_idx, bottom_idx, left_idx, right_idx):
        """Eye Aspect Ratio hesaplama - Basitleştirilmiş versiyon"""
        try:
            if len(landmarks) < 468:
                return 0.3
            
            # Gözün üst ve alt noktaları arasındaki mesafe (dikey)
            vertical = np.linalg.norm(
                np.array([landmarks[top_idx].x, landmarks[top_idx].y]) -
                np.array([landmarks[bottom_idx].x, landmarks[bottom_idx].y])
            )
            
            # Gözün sol ve sağ noktaları arasındaki mesafe (yatay)
            horizontal = np.linalg.norm(
                np.array([landmarks[left_idx].x, landmarks[left_idx].y]) -
                np.array([landmarks[right_idx].x, landmarks[right_idx].y])
            )
            
            if horizontal == 0:
                return 0.3
            
            ear = vertical / horizontal
            return max(0.0, min(1.0, ear))
        except (IndexError, AttributeError) as e:
            return 0.3
    
    def calculate_head_pose(self, landmarks, image_shape):
        """Kafa pozisyonu hesaplama - İyileştirilmiş versiyon"""
        try:
            if len(landmarks) < 468:
                return np.array([0, 0, 0])
            
            # Daha doğru landmark noktaları kullan
            # MediaPipe Face Mesh'te daha stabil noktalar
            image_points = np.array([
                [landmarks[1].x * image_shape[1], landmarks[1].y * image_shape[0]],   # Burun kökü (daha stabil)
                [landmarks[33].x * image_shape[1], landmarks[33].y * image_shape[0]],  # Sol göz dış köşe
                [landmarks[263].x * image_shape[1], landmarks[263].y * image_shape[0]],  # Sağ göz dış köşe
                [landmarks[61].x * image_shape[1], landmarks[61].y * image_shape[0]],  # Sol ağız köşesi
                [landmarks[291].x * image_shape[1], landmarks[291].y * image_shape[0]],  # Sağ ağız köşesi
                [landmarks[4].x * image_shape[1], landmarks[4].y * image_shape[0]]   # Burun ucu
            ], dtype=np.float64)
            
            # 3D model noktaları (güncellenmiş)
            model_points = np.array([
                [0.0, 0.0, 0.0],           # Burun kökü
                [-225.0, 170.0, -135.0],   # Sol göz
                [225.0, 170.0, -135.0],    # Sağ göz
                [-150.0, -150.0, -125.0],  # Sol ağız köşesi
                [150.0, -150.0, -125.0],   # Sağ ağız köşesi
                [0.0, -330.0, -65.0]       # Burun ucu
            ], dtype=np.float64)
            
            # Kamera parametreleri (tahmini)
            focal_length = image_shape[1]
            center = (image_shape[1] / 2, image_shape[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            
            # SolvePnP ile kafa pozisyonu
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Euler açılarına dönüştür
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Doğru Euler açıları hesaplama
                sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
                singular = sy < 1e-6
                
                if not singular:
                    x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi  # Roll
                    y = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi  # Pitch
                    z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi  # Yaw
                else:
                    x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]) * 180 / np.pi
                    y = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi
                    z = 0
                
                angles = np.array([y, z, x])  # Pitch, Yaw, Roll sırası
                return angles
            
            return np.array([0, 0, 0])
        except (IndexError, AttributeError, cv2.error) as e:
            return np.array([0, 0, 0])
    
    def process_frame(self, frame):
        """Frame'i işle ve dikkat skorunu hesapla"""
        try:
            if frame is None or frame.size == 0:
                return self.attention_score, None, None
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Yüz tespit edilemediyse dikkat dağınık say (ama çok agresif değil)
            if not results.multi_face_landmarks:
                if self.calibrated:
                    # Yüz görünmüyorsa (başka yöne çevrilmiş olabilir) dikkat skorunu yavaşça düşür
                    self.attention_score = max(0.0, self.attention_score - 0.05)
                return self.attention_score, None, None
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Göz açıklığı hesaplama
                left_ear = self.calculate_ear(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
                right_ear = self.calculate_ear(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Kafa pozisyonu
                head_pose = self.calculate_head_pose(landmarks, frame.shape)
                
                # Kalibrasyon (ilk birkaç frame)
                if not self.calibrated:
                    self.eye_history.append(avg_ear)
                    self.head_pose_history.append(head_pose.copy())
                    
                    if len(self.eye_history) >= 30:
                        self.base_eye_ratio = np.mean(self.eye_history)
                        self.base_head_pose = np.mean(self.head_pose_history, axis=0)
                        self.calibrated = True
                else:
                    # Dikkat skoru hesaplama - Dengeli ve hassas versiyon
                    # Göz açıklığı kontrolü (gözler kapanıyorsa dikkat dağınık)
                    eye_ratio = avg_ear / self.base_eye_ratio if self.base_eye_ratio > 0 else 1.0
                    
                    # Göz skoru - göz kapalıyken agresif ceza
                    # Gözler kapalıyken (eye_ratio < 0.5) çok düşük puan
                    if eye_ratio >= 0.70:
                        eye_score = 1.0  # Tam açık
                    elif eye_ratio >= 0.60:
                        # 0.60-0.70 arası: 0.9'dan 1.0'a geçiş
                        eye_score = 0.9 + (eye_ratio - 0.60) * 1.0
                    elif eye_ratio >= 0.50:
                        # 0.50-0.60 arası: 0.7'den 0.9'a geçiş
                        eye_score = 0.7 + (eye_ratio - 0.50) * 2.0
                    elif eye_ratio >= 0.40:
                        # 0.40-0.50 arası: 0.4'den 0.7'ye geçiş (yarı açık)
                        eye_score = 0.4 + (eye_ratio - 0.40) * 3.0
                    elif eye_ratio >= 0.25:
                        # 0.25-0.40 arası: 0.1'den 0.4'e geçiş (kısmen kapalı)
                        eye_score = 0.1 + (eye_ratio - 0.25) * 2.0
                    else:
                        # %25'den az: çok düşük puan (kapalı)
                        eye_score = max(0.0, eye_ratio / 0.25) * 0.1
                    
                    # Kafa hareketi kontrolü - Yaw (sağa-sola) öncelikli
                    yaw_diff = abs(head_pose[1] - self.base_head_pose[1])  # Yaw en önemli
                    pitch_diff = abs(head_pose[0] - self.base_head_pose[0])
                    roll_diff = abs(head_pose[2] - self.base_head_pose[2])
                    
                    # Yaw (sağa-sola dönüş) için hassas kontrol - 8 dereceden itibaren ceza
                    if yaw_diff <= 8:
                        yaw_score = 1.0
                    elif yaw_diff <= 15:
                        # 8-15 arası: yumuşak düşüş
                        yaw_score = 1.0 - (yaw_diff - 8) * 0.1
                    elif yaw_diff <= 25:
                        # 15-25 arası: orta düşüş
                        yaw_score = 0.3 - (yaw_diff - 15) * 0.02
                    else:
                        # 25+ derece: çok düşük puan
                        yaw_score = max(0.0, 0.1 - (yaw_diff - 25) * 0.01)
                    
                    # Pitch (yukarı-aşağı) kontrolü - 12 dereceden itibaren ceza
                    if pitch_diff <= 12:
                        pitch_score = 1.0
                    elif pitch_diff <= 20:
                        pitch_score = 1.0 - (pitch_diff - 12) * 0.1
                    else:
                        pitch_score = max(0.0, 0.2 - (pitch_diff - 20) * 0.02)
                    
                    # Roll (eğilme) kontrolü - daha toleranslı
                    if roll_diff <= 15:
                        roll_score = 1.0
                    else:
                        roll_score = max(0.0, 1.0 - (roll_diff - 15) * 0.05)
                    
                    # Kafa skoru - Yaw'a çok daha fazla ağırlık ver
                    head_score = (yaw_score * 0.6 + pitch_score * 0.25 + roll_score * 0.15)
                    
                    # Genel dikkat skoru - göz daha ağırlıklı (göz kapalıyken kesin düşsün)
                    raw_score = (eye_score * 0.65 + head_score * 0.35)
                    
                    # Kalibrasyon sonrası ilk birkaç frame için minimum skor garantisi
                    # Ama göz kapalıyken (eye_score < 0.3) garantiyi uygulama
                    frames_since_calibration = len(self.eye_history) - 30 if len(self.eye_history) > 30 else 0
                    if frames_since_calibration < 30 and eye_score >= 0.3:  # İlk 1 saniye ve göz açıksa
                        # İlk saniyede minimum 0.75 skor garantisi (sadece göz açıksa)
                        raw_score = max(raw_score, 0.75)
                    
                    # Smoothing - ani değişiklikleri yumuşat (daha agresif smoothing)
                    self.attention_history.append(raw_score)
                    if len(self.attention_history) >= 3:
                        # Son 8 frame'in ağırlıklı ortalaması (son frame'ler daha önemli)
                        recent_scores = list(self.attention_history)[-8:]
                        weights = [0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0][-len(recent_scores):]
                        weighted_sum = sum(s * w for s, w in zip(recent_scores, weights))
                        weight_sum = sum(weights)
                        self.attention_score = weighted_sum / weight_sum
                    else:
                        self.attention_score = raw_score
                    
                    self.attention_score = max(0.0, min(1.0, self.attention_score))
                    
                    # Geçmişe ekle
                    self.eye_history.append(avg_ear)
                    self.head_pose_history.append(head_pose.copy())
            
            # Burun ucu pozisyonunu döndür (görselleştirme için)
            nose_tip = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                nose_tip = (landmarks[4].x, landmarks[4].y)  # Burun ucu landmark'ı
            
            return self.attention_score, results.multi_face_landmarks, nose_tip
        except Exception as e:
            print(f"Frame işleme hatası: {e}")
            return self.attention_score, None


class GameManager:
    """Ana oyun yöneticisi - menü ve oyun seçimi"""
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Focus Game - Dikkat Takibi")
        self.clock = pygame.time.Clock()
        self.running = True
        self.bg_color = (20, 30, 40)
        
    def show_main_menu(self):
        """Ana menü - oyun seçimi"""
        title_font = pygame.font.Font(None, 64)
        button_font = pygame.font.Font(None, 48)
        info_font = pygame.font.Font(None, 24)
        
        plane_button_rect = pygame.Rect(self.width // 2 - 150, self.height // 2 - 50, 300, 60)
        circle_button_rect = pygame.Rect(self.width // 2 - 150, self.height // 2 + 30, 300, 60)
        button_color = (50, 150, 50)
        button_hover_color = (70, 170, 70)
        
        selected_game = None
        
        while selected_game is None and self.running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_clicked = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        return None
                    elif event.key == pygame.K_1:
                        selected_game = "plane"
                    elif event.key == pygame.K_2:
                        selected_game = "circle"
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_clicked = True
            
            # Buton hover efektleri
            plane_color = button_hover_color if plane_button_rect.collidepoint(mouse_pos) else button_color
            circle_color = button_hover_color if circle_button_rect.collidepoint(mouse_pos) else button_color
            
            # Buton tıklamaları
            if mouse_clicked:
                if plane_button_rect.collidepoint(mouse_pos):
                    selected_game = "plane"
                elif circle_button_rect.collidepoint(mouse_pos):
                    selected_game = "circle"
            
            # Ekranı temizle
            self.screen.fill(self.bg_color)
            
            # Başlık
            title_text = title_font.render("FOCUS GAME", True, (255, 255, 255))
            title_rect = title_text.get_rect(center=(self.width // 2, 100))
            self.screen.blit(title_text, title_rect)
            
            # Alt başlık
            subtitle_text = info_font.render("Oyun Seçin", True, (200, 200, 200))
            subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, 150))
            self.screen.blit(subtitle_text, subtitle_rect)
            
            # Uçak Oyunu butonu
            pygame.draw.rect(self.screen, plane_color, plane_button_rect, border_radius=10)
            pygame.draw.rect(self.screen, (255, 255, 255), plane_button_rect, width=2, border_radius=10)
            plane_text = button_font.render("1. Uçak Oyunu", True, (255, 255, 255))
            plane_text_rect = plane_text.get_rect(center=plane_button_rect.center)
            self.screen.blit(plane_text, plane_text_rect)
            
            # Daire Oyunu butonu
            pygame.draw.rect(self.screen, circle_color, circle_button_rect, border_radius=10)
            pygame.draw.rect(self.screen, (255, 255, 255), circle_button_rect, width=2, border_radius=10)
            circle_text = button_font.render("2. Daire Oyunu", True, (255, 255, 255))
            circle_text_rect = circle_text.get_rect(center=circle_button_rect.center)
            self.screen.blit(circle_text, circle_text_rect)
            
            # Talimatlar
            inst_text = info_font.render("ESC ile çıkış | Geri tuşu ile menüye dönüş", True, (150, 150, 150))
            inst_rect = inst_text.get_rect(center=(self.width // 2, self.height - 30))
            self.screen.blit(inst_text, inst_rect)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        return selected_game
    
    def run(self):
        """Ana döngü"""
        while self.running:
            selected = self.show_main_menu()
            if not self.running or selected is None:
                break
            
            if selected == "plane":
                plane_game = PlaneGame(self.screen, self.width, self.height, self.clock)
                if not plane_game.run():
                    continue  # Menüye dön
            elif selected == "circle":
                circle_game = CircleGame(self.screen, self.width, self.height, self.clock)
                if not circle_game.run():
                    continue  # Menüye dön
        
        pygame.quit()
        cv2.destroyAllWindows()
        print("Uygulama kapatıldı.")


class PlaneGame:
    def __init__(self, screen, width, height, clock):
        self.screen = screen
        self.width = width
        self.height = height
        self.clock = clock
        
        # Uçak pozisyonu
        self.plane_x = self.width // 2
        self.plane_y = self.height // 2
        self.plane_velocity = 0
        self.max_velocity = 5
        
        # Oyun durumu
        self.running = True
        self.attention_tracker = AttentionTracker()
        self.cap = None
        
        # Renkler
        self.bg_color = (20, 30, 40)
        self.plane_color = (100, 200, 255)
        self.cloud_color = (200, 200, 200)
        
        # Bulutlar
        self.clouds = []
        for i in range(5):
            self.clouds.append({
                'x': np.random.randint(0, self.width),
                'y': np.random.randint(0, self.height),
                'size': np.random.randint(30, 60)
            })
    
    def init_camera(self):
        """Kamerayı başlat"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Kamera açılamadı!")
            return False
        return True
    
    def draw_plane(self, x, y, angle):
        """Uçağı çiz - Güzel ve detaylı versiyon"""
        # Uçak gövdesi (ana gövde)
        body_points = [
            (x, y - 20),      # Burun
            (x - 8, y - 5),   # Sol üst
            (x - 12, y + 8),  # Sol alt
            (x - 8, y + 15),  # Sol kanat alt
            (x, y + 12),      # Alt merkez
            (x + 8, y + 15),  # Sağ kanat alt
            (x + 12, y + 8),  # Sağ alt
            (x + 8, y - 5)    # Sağ üst
        ]
        pygame.draw.polygon(self.screen, (120, 180, 220), body_points)
        pygame.draw.polygon(self.screen, (80, 150, 200), body_points, width=2)
        
        # Kanatlar (yan kanatlar)
        left_wing = [
            (x - 12, y + 2),
            (x - 20, y + 5),
            (x - 18, y + 8),
            (x - 12, y + 8)
        ]
        right_wing = [
            (x + 12, y + 2),
            (x + 20, y + 5),
            (x + 18, y + 8),
            (x + 12, y + 8)
        ]
        pygame.draw.polygon(self.screen, (100, 160, 240), left_wing)
        pygame.draw.polygon(self.screen, (100, 160, 240), right_wing)
        
        # Kuyruk
        tail = [
            (x, y + 12),
            (x - 4, y + 18),
            (x, y + 20),
            (x + 4, y + 18)
        ]
        pygame.draw.polygon(self.screen, (90, 140, 190), tail)
        
        # Kokpit (cam)
        pygame.draw.circle(self.screen, (150, 200, 255), (x, y - 8), 5)
        pygame.draw.circle(self.screen, (200, 230, 255), (x, y - 8), 3)
        
        # Motor/egzoz (arka)
        pygame.draw.ellipse(self.screen, (60, 100, 150), (x - 3, y + 10, 6, 4))
        
        # Detay çizgileri
        pygame.draw.line(self.screen, (70, 120, 170), (x - 8, y - 2), (x + 8, y - 2), 1)
        pygame.draw.line(self.screen, (70, 120, 170), (x - 10, y + 5), (x + 10, y + 5), 1)
    
    def draw_clouds(self):
        """Bulutları çiz"""
        for cloud in self.clouds:
            pygame.draw.circle(self.screen, self.cloud_color, 
                            (cloud['x'], cloud['y']), cloud['size'])
            pygame.draw.circle(self.screen, self.cloud_color, 
                            (cloud['x'] + 20, cloud['y']), cloud['size'] - 10)
            pygame.draw.circle(self.screen, self.cloud_color, 
                            (cloud['x'] - 20, cloud['y']), cloud['size'] - 10)
    
    def update_clouds(self):
        """Bulutları hareket ettir"""
        for cloud in self.clouds:
            cloud['x'] -= 1
            if cloud['x'] < -100:
                cloud['x'] = self.width + 100
                cloud['y'] = np.random.randint(0, self.height)
    
    def run(self):
        """Ana oyun döngüsü"""
        try:
            if not self.init_camera():
                return False
            
            font = pygame.font.Font(None, 36)
            small_font = pygame.font.Font(None, 24)
            
            self.plane_y = self.height // 2
            self.plane_velocity = 0
            self.attention_tracker = AttentionTracker()
            
            # --- YENİ AYARLAR: GÜVENLİ KUTU ---
            CEILING_LIMIT = 100 
            
            # Kutu Boyutları
            SAFE_ZONE_WIDTH = 200   # Genişlik
            SAFE_ZONE_HEIGHT = 400  # Yükseklik (Dikey alan)
            
            # Kutunun Koordinatlarını Hesapla (Ekranın tam ortası)
            # Örn: 800x600 ekranda -> X: 300-500 arası, Y: 100-500 arası
            safe_zone_left = (self.width - SAFE_ZONE_WIDTH) // 2
            safe_zone_right = safe_zone_left + SAFE_ZONE_WIDTH
            
            safe_zone_top = (self.height - SAFE_ZONE_HEIGHT) // 2
            safe_zone_bottom = safe_zone_top + SAFE_ZONE_HEIGHT
            
            while self.running:
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                return False
                            elif event.key == pygame.K_BACKSPACE:
                                return True
                    
                    ret, frame = self.cap.read()
                    if not ret: continue
                    
                    attention_score, face_landmarks, nose_tip = self.attention_tracker.process_frame(frame)
                    
                    # --- GÜVENLİ KUTU KONTROLÜ (Hem Yatay Hem Dikey) ---
                    is_in_safe_zone = True
                    
                    if nose_tip is not None:
                        nose_x = int(nose_tip[0] * self.width)
                        nose_y = int(nose_tip[1] * self.height)
                        
                        # Kutu dışına çıktı mı?
                        # Sol-Sağ VEYA Yukarı-Aşağı sınır ihlali var mı?
                        if (nose_x < safe_zone_left or nose_x > safe_zone_right or 
                            nose_y < safe_zone_top or nose_y > safe_zone_bottom):
                            
                            is_in_safe_zone = False
                            # Cezalandır
                            attention_score -= 0.5 
                            attention_score = max(0.0, attention_score)
                    
                    # Uçak hareketi
                    if attention_score > 0.7:
                        self.plane_velocity = max(-self.max_velocity, self.plane_velocity - 0.2)
                    else:
                        self.plane_velocity = min(self.max_velocity, self.plane_velocity + 0.3)
                    
                    self.plane_y += self.plane_velocity
                    
                    if self.plane_y < CEILING_LIMIT:
                        self.plane_y = CEILING_LIMIT
                    elif self.plane_y > self.height:
                        self.plane_y = self.height
                    
                    # --- ÇİZİM ---
                    self.screen.fill(self.bg_color)
                    
                    # Güvenli Kutuyu Çiz (Kullanıcı sınırlarını görsün)
                    # Gri renkte, içi boş bir dikdörtgen
                    box_color = (50, 60, 70)
                    pygame.draw.rect(self.screen, box_color, 
                                   (safe_zone_left, safe_zone_top, SAFE_ZONE_WIDTH, SAFE_ZONE_HEIGHT), 2)
                    
                    self.draw_clouds()
                    self.update_clouds()
                    
                    angle = -self.plane_velocity * 5
                    self.draw_plane(self.plane_x, int(self.plane_y), angle)
                    
                    # Skor
                    attention_text = f"Dikkat: {attention_score:.2f}"
                    color = (0, 255, 0) if attention_score > 0.7 else (255, 0, 0)
                    text_surface = font.render(attention_text, True, color)
                    self.screen.blit(text_surface, (10, 10))
                    
                    if not self.attention_tracker.calibrated:
                        calib_text = small_font.render("Kalibrasyon yapılıyor...", True, (255, 255, 255))
                        self.screen.blit(calib_text, (10, 50))
                    
                    back_text = small_font.render("Geri: Backspace | Çıkış: ESC", True, (150, 150, 150))
                    self.screen.blit(back_text, (10, self.height - 30))
                    
                    # --- BURUN POINTER & UYARI ---
                    if nose_tip is not None and self.attention_tracker.calibrated:
                        pointer_color = (0, 255, 255) if is_in_safe_zone else (255, 0, 0)
                        
                        pygame.draw.circle(self.screen, pointer_color, (nose_x, nose_y), 8, 2)
                        pygame.draw.circle(self.screen, pointer_color, (nose_x, nose_y), 3)
                        
                        if not is_in_safe_zone:
                            # Hangi yöne gitmesi gerektiğini söyleyelim
                            msg = "MERKEZE DON!"
                            if nose_y < safe_zone_top: msg = "ASAGI BAK!"
                            elif nose_y > safe_zone_bottom: msg = "YUKARI BAK!"
                            
                            warn_text = small_font.render(msg, True, (255, 50, 50))
                            self.screen.blit(warn_text, (nose_x - 40, nose_y - 30))

                    if face_landmarks is None:
                        no_face_text = small_font.render("Yüz Bulunamadı", True, (255, 200, 0))
                        self.screen.blit(no_face_text, (10, 80))
                    
                    if self.plane_y >= self.height - 20:
                        game_over_text = font.render("Dikkat Dağınık! Oyun Bitti", True, (255, 0, 0))
                        text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2))
                        self.screen.blit(game_over_text, text_rect)
                    
                    pygame.display.flip()
                    self.clock.tick(30)
                    
                except Exception as e:
                    print(f"Hata: {e}")
                    return False
        
        except Exception as e:
            print(f"Kritik Hata: {e}")
            return False
        finally:
            if self.cap: self.cap.release()
        return False


class CircleGame:
    """Daire oyunu - ikişer atladığında boşluk tuşuna bas"""
    def __init__(self, screen, width, height, clock):
        self.screen = screen
        self.width = width
        self.height = height
        self.clock = clock
        
        # Oyun durumu
        self.running = True
        self.bg_color = (20, 30, 40)
        self.circle_color = (100, 200, 255)
        self.active_color = (255, 200, 50)  # Tek renk - sarı
        
        # Daireler
        self.num_circles = 12
        self.circle_radius = 30
        self.center_x = width // 2
        self.center_y = height // 2
        self.circle_radius_distance = 150
        
        # Oyun mantığı
        self.current_index = 0
        self.jump_interval = 1.0  # Saniye - yavaşlatıldı
        self.last_jump_time = time.time()
        self.jump_count = 0  # Kaç daire atlandı (1 veya 2)
        self.score = 0
        
        # İkişer atlama kontrolü
        self.is_double_jump = False
        self.double_jump_chance = 0.3  # %30 şans
        self.double_jump_start_time = None
        self.double_jump_timeout = 2.0  # 2 saniye süre
        self.missed_message_time = None
        self.missed_message_duration = 1.5  # "Kaçırdın" mesajı 1.5 saniye gösterilsin
        self.wrong_press_time = None
        self.wrong_press_duration = 1.5  # "Yanlış bastınız" mesajı 1.5 saniye gösterilsin
        
        # Zamanlayıcı
        self.game_duration = 180  # 3 dakika = 180 saniye
        self.start_time = time.time()
        
    def run(self):
        """Ana oyun döngüsü"""
        font = pygame.font.Font(None, 48)
        small_font = pygame.font.Font(None, 24)
        large_font = pygame.font.Font(None, 72)
        
        self.start_time = time.time()
        
        while self.running:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            remaining_time = max(0, self.game_duration - elapsed_time)
            
            if remaining_time <= 0:
                self.running = False
                break
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_BACKSPACE:
                        return True
                    elif event.key == pygame.K_SPACE:
                        # Boşluğa basıldı
                        if self.is_double_jump and self.double_jump_start_time is not None:
                            # Çift atlama sırasında basıldı mı?
                            time_since_double = current_time - self.double_jump_start_time
                            if time_since_double <= self.double_jump_timeout:
                                # ZAMANINDA BASILDI!
                                self.score += 10
                                self.is_double_jump = False
                                self.double_jump_start_time = None
                            # Zaman geçtiyse zaten aşağıdaki döngüde "kaçırdın" mesajı verilecek
                        else:
                            # Yanlış zamanda basıldı (Tek atlama sırasında)
                            self.wrong_press_time = current_time
            
            # Zaman kontrolü ve Atlama Mantığı
            if current_time - self.last_jump_time >= self.jump_interval:
                self.last_jump_time = current_time
                
                # --- YENİ DÜZENLEME: KAÇIRMA KONTROLÜ ---
                # Eğer yeni bir atlama sırası geldiyse ve hala bir önceki çift atlama (double jump)
                # aktifse, demek ki kullanıcı boşluğa basmayı unuttu veya kaçırdı.
                if self.is_double_jump:
                     self.missed_message_time = current_time # Mesajı tetikle
                     self.is_double_jump = False # Durumu sıfırla
                     self.double_jump_start_time = None

                # Yeni atlama türünü belirle (Tek mi Çift mi?)
                if np.random.random() < self.double_jump_chance:
                    self.jump_count = 2
                    self.is_double_jump = True
                    self.double_jump_start_time = current_time # Süreyi başlat
                else:
                    self.jump_count = 1
                    self.is_double_jump = False
                    self.double_jump_start_time = None
                
                self.current_index = (self.current_index + self.jump_count) % self.num_circles
            
            # Ekranı temizle
            self.screen.fill(self.bg_color)
            
            # Daireleri Çiz
            for i in range(self.num_circles):
                angle = (2 * np.pi * i) / self.num_circles
                x = self.center_x + self.circle_radius_distance * np.cos(angle)
                y = self.center_y + self.circle_radius_distance * np.sin(angle)
                
                if i == self.current_index:
                    color = self.active_color
                    pygame.draw.circle(self.screen, color, (int(x), int(y)), self.circle_radius)
                    pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), self.circle_radius, 3)
                else:
                    pygame.draw.circle(self.screen, self.circle_color, (int(x), int(y)), self.circle_radius)
                    pygame.draw.circle(self.screen, (150, 150, 150), (int(x), int(y)), self.circle_radius, 2)
            
            # Süre Yazısı
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            time_text = font.render(f"Süre: {minutes:02d}:{seconds:02d}", True, (255, 255, 255))
            self.screen.blit(time_text, (10, 10))
            
            # --- MESAJLAR ---
            
            # "KAÇIRDIN!" Mesajı
            if self.missed_message_time is not None:
                if current_time - self.missed_message_time < self.missed_message_duration:
                    missed_text = large_font.render("KAÇIRDIN!", True, (255, 50, 50)) # Kırmızı
                    missed_rect = missed_text.get_rect(center=(self.width // 2, self.height // 2))
                    self.screen.blit(missed_text, missed_rect)
                else:
                    self.missed_message_time = None
            
            # "YANLIŞ BASTINIZ!" Mesajı
            if self.wrong_press_time is not None:
                if current_time - self.wrong_press_time < self.wrong_press_duration:
                    wrong_text = large_font.render("YANLIŞ!", True, (255, 150, 0)) # Turuncu
                    wrong_rect = wrong_text.get_rect(center=(self.width // 2, self.height // 2))
                    self.screen.blit(wrong_text, wrong_rect)
                else:
                    self.wrong_press_time = None
            
            # Alt Bilgiler
            inst_text = small_font.render("2 birim atlarsa BOŞLUK tuşuna bas", True, (200, 200, 200))
            inst_rect = inst_text.get_rect(center=(self.width // 2, self.height - 50))
            self.screen.blit(inst_text, inst_rect)
            
            back_text = small_font.render("Geri: Backspace | Çıkış: ESC", True, (150, 150, 150))
            back_rect = back_text.get_rect(center=(self.width // 2, self.height - 20))
            self.screen.blit(back_text, back_rect)
            
            # Oyun Sonu
            if not self.running:
                game_over_text = font.render("Süre Doldu!", True, (255, 200, 0))
                text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
                self.screen.blit(game_over_text, text_rect)
                
                final_score_text = font.render(f"Final Skor: {self.score}", True, (255, 255, 255))
                score_rect = final_score_text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(final_score_text, score_rect)
                
                pygame.display.flip()
                pygame.time.wait(3000)
                return True
            
            pygame.display.flip()
            self.clock.tick(60)
        
        return True  # Menüye dön


if __name__ == "__main__":
    try:
        manager = GameManager()
        manager.run()
    except Exception as e:
        print(f"Program başlatma hatası: {e}")
        import traceback
        traceback.print_exc()
        input("Çıkmak için Enter'a basın...")

