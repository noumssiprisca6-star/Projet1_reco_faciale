import cv2
import sys
import os

def main():
    # Chemins vers les modèles YuNet et SFace (placés dans le dossier 'models/' à l'intérieur de 'src/')
    detector_model ="../models/yunet.onnx" 
    recognizer_model = "../models/face_recognition_sface_2021dec.onnx"

    if not os.path.exists(detector_model) or not os.path.exists(recognizer_model):
        print("Erreur : Les modèles ONNX de YuNet et SFace sont introuvables dans le dossier 'models/' situé dans 'src/'.")
        return

    # Initialisation du détecteur de visage YuNet
    detector = cv2.FaceDetectorYN.create(
        detector_model,
        "",
        (320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )

    # Initialisation du reconnaisseur SFace
    recognizer = cv2.FaceRecognizerSF.create(
        recognizer_model,
        ""
    )

    # Chargement de la base de données préparée par Bias (dossier 'dataset/' à côté de 'src/' ou dedans)
    # Ajustez le chemin 'dataset' si celui-ci se trouve à la racine du projet (ex: '../dataset')
    dataset_path = "../data" 
    if not os.path.exists(dataset_path):
        dataset_path = "data" # S'il est directement dans src/

    database_embeddings = {}

    if os.path.exists(dataset_path):
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_dir):
                embeddings_list = []
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        detector.setInputSize((img.shape[1], img.shape[0]))
                        _, faces = detector.detect(img)
                        if faces is not None and len(faces) > 0:
                            face_aligned = recognizer.alignCrop(img, faces[0])
                            feature = recognizer.feature(face_aligned)
                            embeddings_list.append(feature)
                if embeddings_list:
                    database_embeddings[person_name] = embeddings_list

    # Ouverture de la webcam
    cap = cv2.VideoCapture(0)
    
    print("Démarrage de l'application. Appuyez sur 'q' pour quitter.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
                cosine_threshold = 0.363 

                for name, ref_list in database_embeddings.items():
                    for ref_feature in ref_list:
                        similarity = recognizer.match(feature, ref_feature, cv2.FaceRecognizerSF_FR_COSINE)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            if similarity >= cosine_threshold:
                                identity = name

                color = (0, 255, 0) if identity != "Inconnu" else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
                
                label = f"{identity} ({max_similarity:.2f})"
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Reconnaissance Faciale - Projet 1", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()