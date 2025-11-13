from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(model_name, model, features_train, target_train, features_test, target_test):
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)

    accuracy = accuracy_score(target_test, predictions)
    precision = precision_score(target_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(target_test, predictions, average="weighted", zero_division=0)

    print(f"--- {model_name} ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print()
    
    return accuracy, precision, recall, predictions
