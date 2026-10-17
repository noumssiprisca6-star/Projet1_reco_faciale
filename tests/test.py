import cv2
import os
import sys

def load_database(dataset_path, detector, recognizer):
    database_embeddings = {}
    if not os.path.exists(dataset_path):
        print(f"[AVERTISSEMENT] Le dossier dataset '{dataset_path}' est introuvable.")
        return database_embeddings

    print("[INFO] Chargement et encodage de la base de données des étudiants...")
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            embeddings_list = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    h, w, _ = img.shape
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(img)
                    if faces is not None and len(faces) > 0:
                        face_aligned = recognizer.alignCrop(img, faces[0])
                        feature = recognizer.feature(face_aligned)
                        embeddings_list.append(feature)
            if embeddings_list:
                database_embeddings[person_name] = embeddings_list
                print(f"  -> {person_name} : {len(embeddings_list)} images encodées.")
                
    print("[INFO] Fin du chargement de la base.\n")
    return database_embeddings

def main():
    # 1. Chemins des modèles (en partant de 'src/' vers '../models/')
    detector_model = "../models/yunet.onnx"
    recognizer_model = "../models/face_recognition_sface_2021dec.onnx"

    if not os.path.exists(detector_model) or not os.path.exists(recognizer_model):
        print("Erreur : Les fichiers .onnx sont introuvables dans le dossier '../models/'.")
        return

    # 2. Initialisation des modèles YuNet et SFace
    detector = cv2.FaceDetectorYN.create(detector_model, "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(recognizer_model, "")

    # 3. Chargement de la base de données de Bias
    dataset_path = "../data/raw"
    database_embeddings = load_database(dataset_path, detector, recognizer)

    # 4. TEST AVEC UNE IMAGE FIXE (Idéal pour tester le travail des camarades)
    # Mettez une photo de test dans un dossier '../test/' et indiquez son nom ici :
    image_test_path = "../data/raw/aristide/WhatsApp Image 2026-10-14 at 09.24.33 (1).jpeg"  
    
    if not os.path.exists(image_test_path):
        print(f"Erreur : L'image de test '{image_test_path}' est introuvable. Veuillez y placer une photo.")
        return

    frame = cv2.imread(image_test_path)
    h, w, _ = frame.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            box = face[0:4].astype(int)
            confidence_score = face[-1]

            face_aligned = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(face_aligned)

            identity = "Inconnu"
            max_similarity = 0.0
            cosine_threshold = 0.363  # Seuil de similarité SFace

            for name, ref_list in database_embeddings.items():
                for ref_feature in ref_list:
                    similarity = recognizer.match(feature, ref_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        if similarity >= cosine_threshold:
                            identity = name

            # Affichage du résultat (Vert si reconnu, Rouge si Inconnu)
            color = (0, 255, 0) if identity != "Inconnu" else (0, 0, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
            
            label = f"{identity} ({max_similarity:.2f})"
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        print("Aucun visage détecté sur cette image.")

    # 5. Affichage de la fenêtre de test
    print("Affichage du résultat. Appuyez sur n'importe quelle touche du clavier pour fermer la fenêtre.")
    cv2.imshow("Test - Reconnaissance Faciale (Projet 1)", frame)
    
    cv2.waitKey(0)  # Attend l'appui sur une touche pour fermer proprement
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()