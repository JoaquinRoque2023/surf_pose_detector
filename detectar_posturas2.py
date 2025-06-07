import cv2
import mediapipe as mp
import numpy as np
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque
import math

@dataclass
class ManeuverDetection:
    """Clase para almacenar detecciones de maniobras"""
    maneuver_type: str
    confidence: float
    timestamp: float
    frame_number: int
    body_position: Dict[str, Tuple[float, float]]
    board_angle: float

class SurfManeuverDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Buffer para análisis temporal de poses
        self.pose_history = deque(maxlen=30)  # ~1 segundo a 30fps
        self.detections = []
        
        # Parámetros para cada maniobra
        self.maneuver_thresholds = {
            'take_off': {
                'min_knee_bend': 45,    # ángulo mínimo de flexión de rodillas
                'max_torso_angle': 30,  # inclinación máxima del torso
                'duration_frames': 15   # frames mínimos para confirmar
            },
            'bottom_turn': {
                'min_shoulder_rotation': 35,
                'min_hip_rotation': 25,
                'velocity_threshold': 0.1
            },
            'top_turn': {
                'min_shoulder_rotation': 40,
                'vertical_position_min': 0.3,  # parte alta de la ola
                'centrifugal_force': True
            },
            'cutback': {
                'direction_change': True,
                'min_rotation': 90,
                'return_to_power': True
            },
            'floater': {
                'vertical_velocity': 0.05,
                'foam_proximity': True,
                'horizontal_movement': True
            },
            'kick_out': {
                'exit_velocity': 0.08,
                'direction_vector': 'backward',
                'controlled_exit': True
            }
        }

        # Diccionario para nombres en español
        self.maneuver_names = {
            'take_off': 'Take Off',
            'bottom_turn': 'Bottom Turn',
            'top_turn': 'Top Turn',
            'cutback': 'Cutback',
            'floater': 'Floater',
            'kick_out': 'Kick Out'
        }

    def calculate_joint_angle(self, p1, p2, p3):
        """Calcula el ángulo entre tres puntos (articulaciones)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def extract_pose_features(self, landmarks) -> Dict:
        """Extrae características relevantes de la pose"""
        if not landmarks:
            return None
            
        # Puntos clave para surf
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        features = {
            'shoulder_center': ((left_shoulder.x + right_shoulder.x) / 2, 
                              (left_shoulder.y + right_shoulder.y) / 2),
            'hip_center': ((left_hip.x + right_hip.x) / 2, 
                          (left_hip.y + right_hip.y) / 2),
            'knee_angles': {
                'left': self.calculate_joint_angle(
                    (left_hip.x, left_hip.y),
                    (left_knee.x, left_knee.y),
                    (left_ankle.x, left_ankle.y)
                ),
                'right': self.calculate_joint_angle(
                    (right_hip.x, right_hip.y),
                    (right_knee.x, right_knee.y),
                    (right_ankle.x, right_ankle.y)
                )
            },
            'torso_angle': math.atan2(
                left_shoulder.y - left_hip.y,
                left_shoulder.x - left_hip.x
            ) * 180 / math.pi,
            'shoulder_rotation': math.atan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ) * 180 / math.pi
        }
        
        return features

    def detect_take_off(self, current_features, history) -> float:
        """Detecta la maniobra de take off"""
        if len(history) < 10:
            return 0.0
            
        confidence = 0.0
        
        # Verificar transición de acostado a parado
        knee_bend_left = current_features['knee_angles']['left']
        knee_bend_right = current_features['knee_angles']['right']
        avg_knee_bend = (knee_bend_left + knee_bend_right) / 2
        
        # Puntuación basada en flexión de rodillas
        if 30 <= avg_knee_bend <= 90:
            confidence += 0.4
            
        # Verificar estabilidad del torso
        torso_angle = abs(current_features['torso_angle'])
        if torso_angle < 30:
            confidence += 0.3
            
        # Verificar progresión temporal (de horizontal a vertical)
        if len(history) >= 5:
            prev_torso = abs(history[-5]['torso_angle'])
            if prev_torso > torso_angle + 10:  # Progresión hacia vertical
                confidence += 0.3
                
        return min(confidence, 1.0)

    def detect_bottom_turn(self, current_features, history) -> float:
        """Detecta bottom turn"""
        if len(history) < 5:
            return 0.0
            
        confidence = 0.0
        
        # Rotación de hombros pronunciada
        shoulder_rotation = abs(current_features['shoulder_rotation'])
        if shoulder_rotation > 35:
            confidence += 0.4
            
        # Movimiento descendente seguido de ascendente
        if len(history) >= 3:
            current_y = current_features['hip_center'][1]
            prev_y = history[-3]['hip_center'][1]
            
            if current_y < prev_y:  # Movimiento hacia arriba
                confidence += 0.3
                
        # Velocidad de cambio de dirección
        if len(history) >= 2:
            prev_rotation = history[-1]['shoulder_rotation']
            rotation_change = abs(current_features['shoulder_rotation'] - prev_rotation)
            if rotation_change > 15:
                confidence += 0.3
                
        return min(confidence, 1.0)

    def detect_top_turn(self, current_features, history) -> float:
        """Detecta top turn"""
        confidence = 0.0
        
        # Posición vertical alta (parte superior de la ola)
        hip_y = current_features['hip_center'][1]
        if hip_y < 0.4:  # Parte alta del frame
            confidence += 0.3
            
        # Rotación pronunciada
        shoulder_rotation = abs(current_features['shoulder_rotation'])
        if shoulder_rotation > 40:
            confidence += 0.4
            
        # Ángulo del torso indicando giro cerrado
        torso_angle = abs(current_features['torso_angle'])
        if 20 <= torso_angle <= 70:
            confidence += 0.3
            
        return min(confidence, 1.0)

    def detect_cutback(self, current_features, history) -> float:
        """Detecta cutback"""
        if len(history) < 10:
            return 0.0
            
        confidence = 0.0
        
        # Cambio de dirección pronunciado
        if len(history) >= 5:
            current_rotation = current_features['shoulder_rotation']
            past_rotation = history[-5]['shoulder_rotation']
            
            rotation_change = abs(current_rotation - past_rotation)
            if rotation_change > 90:
                confidence += 0.5
                
        # Movimiento hacia la espuma (simulado con posición)
        hip_x = current_features['hip_center'][0]
        if len(history) >= 3:
            prev_hip_x = history[-3]['hip_center'][0]
            if abs(hip_x - prev_hip_x) > 0.1:  # Movimiento lateral significativo
                confidence += 0.3
                
        # Patrón de S en el movimiento
        if len(history) >= 8:
            x_positions = [h['hip_center'][0] for h in history[-8:]]
            if self.detect_s_pattern(x_positions):
                confidence += 0.2
                
        return min(confidence, 1.0)

    def detect_s_pattern(self, positions) -> bool:
        """Detecta patrón en S característico del cutback"""
        if len(positions) < 6:
            return False
            
        # Simplificado: detectar cambios de dirección múltiples
        direction_changes = 0
        for i in range(1, len(positions) - 1):
            if ((positions[i] - positions[i-1]) * (positions[i+1] - positions[i])) < 0:
                direction_changes += 1
                
        return direction_changes >= 2

    def detect_floater(self, current_features, history) -> float:
        """Detecta floater"""
        if len(history) < 5:
            return 0.0
            
        confidence = 0.0
        
        # Movimiento horizontal sobre la espuma
        current_y = current_features['hip_center'][1]
        if len(history) >= 3:
            y_positions = [h['hip_center'][1] for h in history[-3:]]
            y_stability = np.std(y_positions)
            
            if y_stability < 0.02 and current_y < 0.5:  # Movimiento horizontal estable
                confidence += 0.4
                
        # Postura extendida
        avg_knee_angle = (current_features['knee_angles']['left'] + 
                         current_features['knee_angles']['right']) / 2
        if avg_knee_angle > 160:  # Piernas extendidas
            confidence += 0.3
            
        # Movimiento lateral
        if len(history) >= 2:
            x_movement = abs(current_features['hip_center'][0] - history[-2]['hip_center'][0])
            if x_movement > 0.05:
                confidence += 0.3
                
        return min(confidence, 1.0)

    def detect_kick_out(self, current_features, history) -> float:
        """Detecta kick out"""
        if len(history) < 8:
            return 0.0
            
        confidence = 0.0
        
        # Movimiento hacia atrás o hacia arriba
        if len(history) >= 5:
            current_pos = current_features['hip_center']
            past_pos = history[-5]['hip_center']
            
            # Movimiento hacia atrás (aumento en Y) o hacia arriba (disminución en Y)
            y_movement = current_pos[1] - past_pos[1]
            if y_movement > 0.1 or y_movement < -0.08:
                confidence += 0.4
                
        # Postura controlada
        torso_angle = abs(current_features['torso_angle'])
        if torso_angle < 25:  # Postura erguida y controlada
            confidence += 0.3
            
        # Velocidad de salida
        if len(history) >= 3:
            positions = [h['hip_center'] for h in history[-3:]] + [current_features['hip_center']]
            velocities = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                velocity = math.sqrt(dx**2 + dy**2)
                velocities.append(velocity)
                
            if len(velocities) > 0 and max(velocities) > 0.08:
                confidence += 0.3
                
        return min(confidence, 1.0)

    def analyze_frame_realtime(self, landmarks, frame_number: int, timestamp: float) -> Dict:
        """Analiza un frame y retorna detecciones para mostrar en tiempo real"""
        features = self.extract_pose_features(landmarks)
        if not features:
            return {}
            
        self.pose_history.append(features)
        
        # Detectar cada tipo de maniobra
        maneuver_confidences = {
            'take_off': self.detect_take_off(features, list(self.pose_history)),
            'bottom_turn': self.detect_bottom_turn(features, list(self.pose_history)),
            'top_turn': self.detect_top_turn(features, list(self.pose_history)),
            'cutback': self.detect_cutback(features, list(self.pose_history)),
            'floater': self.detect_floater(features, list(self.pose_history)),
            'kick_out': self.detect_kick_out(features, list(self.pose_history))
        }
        
        # Filtrar solo las detecciones con confianza > 0.3 para mostrar
        active_detections = {}
        for maneuver, confidence in maneuver_confidences.items():
            if confidence > 0.3:
                active_detections[maneuver] = confidence
        
        return active_detections

    def draw_detections_on_frame(self, image, detections: Dict, landmarks):
        """Dibuja las detecciones en el frame"""
        height, width, _ = image.shape
        
        # Dibujar el esqueleto de la pose
        if landmarks:
            self.mp_drawing.draw_landmarks(
                image, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        # Dibujar las detecciones de maniobras
        y_offset = 50
        for maneuver, confidence in detections.items():
            maneuver_name = self.maneuver_names.get(maneuver, maneuver)
            confidence_percent = int(confidence * 100)
            
            # Determinar color basado en confianza
            if confidence > 0.7:
                color = (0, 255, 0)  # Verde para alta confianza
            elif confidence > 0.5:
                color = (0, 255, 255)  # Amarillo para media confianza
            else:
                color = (255, 255, 0)  # Cyan para baja confianza
            
            # Texto principal
            text = f"{maneuver_name}: {confidence_percent}%"
            cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2, cv2.LINE_AA)
            
            # Barra de confianza
            bar_width = int(200 * confidence)
            cv2.rectangle(image, (20, y_offset + 10), (20 + bar_width, y_offset + 20), color, -1)
            cv2.rectangle(image, (20, y_offset + 10), (220, y_offset + 20), (255, 255, 255), 1)
            
            y_offset += 60
        
        # Información adicional en la esquina inferior
        info_text = f"Frame: {cv2.getTickCount()}"
        cv2.putText(image, info_text, (width - 200, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image

    def process_video_realtime(self, video_path: str):
        """Procesa un video mostrando detecciones en tiempo real"""
        cap = cv2.VideoCapture(video_path)
        
        # Si es 0, usa la cámara web
        if video_path == "0":
            cap = cv2.VideoCapture(0)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_number = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("No se puede leer el frame. Fin del video o error.")
                    break
                
                timestamp = frame_number / fps if fps > 0 else 0
                
                # Procesar frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                
                # Convertir de vuelta a BGR para mostrar
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image.flags.writeable = True
                
                # Analizar y obtener detecciones
                detections = {}
                if results.pose_landmarks:
                    detections = self.analyze_frame_realtime(
                        results.pose_landmarks.landmark, frame_number, timestamp
                    )
                
                # Dibujar detecciones en el frame
                image = self.draw_detections_on_frame(image, detections, results.pose_landmarks)
                
                # Mostrar el frame
                cv2.imshow('Surf Maneuver Detection', image)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' o ESC para salir
                    break
                elif key == ord('p'):  # 'p' para pausar
                    cv2.waitKey(0)
                
                frame_number += 1
        
        cap.release()
        cv2.destroyAllWindows()

    # PARTE DEL JSON COMENTADA - Descomenta para usar
    """
    def analyze_frame(self, landmarks, frame_number: int, timestamp: float):
        features = self.extract_pose_features(landmarks)
        if not features:
            return
            
        self.pose_history.append(features)
        
        # Detectar cada tipo de maniobra
        maneuver_confidences = {
            'take_off': self.detect_take_off(features, list(self.pose_history)),
            'bottom_turn': self.detect_bottom_turn(features, list(self.pose_history)),
            'top_turn': self.detect_top_turn(features, list(self.pose_history)),
            'cutback': self.detect_cutback(features, list(self.pose_history)),
            'floater': self.detect_floater(features, list(self.pose_history)),
            'kick_out': self.detect_kick_out(features, list(self.pose_history))
        }
        
        # Registrar detecciones con confianza > 0.5
        for maneuver, confidence in maneuver_confidences.items():
            if confidence > 0.5:
                detection = ManeuverDetection(
                    maneuver_type=maneuver,
                    confidence=confidence,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    body_position={
                        'shoulder_center': features['shoulder_center'],
                        'hip_center': features['hip_center']
                    },
                    board_angle=features['torso_angle']
                )
                self.detections.append(detection)

    def process_video(self, video_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_number = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                timestamp = frame_number / fps
                
                # Procesar frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                
                if results.pose_landmarks:
                    self.analyze_frame(results.pose_landmarks.landmark, frame_number, timestamp)
                
                frame_number += 1
                
        cap.release()
        
        # Generar JSON de respuesta
        return self.generate_response_json()

    def generate_response_json(self) -> Dict:
        # Agrupar detecciones por tipo
        maneuver_summary = {}
        
        for detection in self.detections:
            maneuver = detection.maneuver_type
            if maneuver not in maneuver_summary:
                maneuver_summary[maneuver] = {
                    'detected_count': 0,
                    'max_confidence': 0.0,
                    'avg_confidence': 0.0,
                    'detections': []
                }
            
            maneuver_summary[maneuver]['detected_count'] += 1
            maneuver_summary[maneuver]['max_confidence'] = max(
                maneuver_summary[maneuver]['max_confidence'], 
                detection.confidence
            )
            maneuver_summary[maneuver]['detections'].append({
                'timestamp': detection.timestamp,
                'confidence': detection.confidence,
                'frame_number': detection.frame_number
            })
        
        # Calcular promedios
        for maneuver in maneuver_summary:
            confidences = [d.confidence for d in self.detections if d.maneuver_type == maneuver]
            maneuver_summary[maneuver]['avg_confidence'] = sum(confidences) / len(confidences)
        
        response = {
            'video_analysis': {
                'total_detections': len(self.detections),
                'maneuvers_found': list(maneuver_summary.keys()),
                'detailed_analysis': maneuver_summary
            },
            'performance_score': self.calculate_overall_score(maneuver_summary),
            'recommendations': self.generate_recommendations(maneuver_summary)
        }
        
        return response

    def calculate_overall_score(self, summary: Dict) -> Dict:
        basic_maneuvers = ['take_off', 'bottom_turn', 'top_turn']
        advanced_maneuvers = ['cutback', 'floater', 'kick_out']
        
        basic_score = 0
        advanced_score = 0
        
        for maneuver in basic_maneuvers:
            if maneuver in summary:
                basic_score += summary[maneuver]['max_confidence']
        
        for maneuver in advanced_maneuvers:
            if maneuver in summary:
                advanced_score += summary[maneuver]['max_confidence']
        
        return {
            'basic_maneuvers_score': (basic_score / len(basic_maneuvers)) * 100,
            'advanced_maneuvers_score': (advanced_score / len(advanced_maneuvers)) * 100,
            'overall_score': ((basic_score + advanced_score) / 6) * 100
        }

    def generate_recommendations(self, summary: Dict) -> List[str]:
        recommendations = []
        
        if 'take_off' not in summary or summary['take_off']['max_confidence'] < 0.7:
            recommendations.append("Practica la transición del take off para mayor fluidez")
        
        if 'bottom_turn' not in summary:
            recommendations.append("Trabaja en tus bottom turns para generar más velocidad")
        
        if 'cutback' not in summary:
            recommendations.append("Intenta incorporar cutbacks para mantener velocidad en la ola")
        
        advanced_count = sum(1 for m in ['cutback', 'floater', 'kick_out'] if m in summary)
        if advanced_count < 2:
            recommendations.append("Experimenta con maniobras más avanzadas")
        
        return recommendations
    """

# Uso del detector
if __name__ == "__main__":
    detector = SurfManeuverDetector()
    
    print("=== SURF MANEUVER DETECTOR ===")
    print("Controles:")
    print("- 'q' o ESC: Salir")
    print("- 'p': Pausar/Reanudar")
    print("- Colores:")
    print("  * Verde: Alta confianza (>70%)")
    print("  * Amarillo: Media confianza (50-70%)")
    print("  * Cyan: Baja confianza (30-50%)")
    print("\nIniciando detección en tiempo real...")
    
    # Para usar con un archivo de video
    video_path = input("Ingresa la ruta del video (o '0' para cámara web): ").strip()
    if video_path == "":
        video_path = "surfista.mp4"  # Video por defecto
    
    # Procesar video en tiempo real
    detector.process_video_realtime(video_path)
    
    # PARA GENERAR JSON - Descomenta las siguientes líneas:
    """
    result = detector.process_video(video_path)
    
    # Guardar resultado en JSON
    with open("surf_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("Análisis completado. Resultados guardados en surf_analysis.json")
    print(json.dumps(result, indent=2))
    """