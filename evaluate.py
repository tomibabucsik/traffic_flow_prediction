import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate(model, test_loader, scaler, criterion=torch.nn.MSELoss()):
    model.eval()
    test_loss = 0
    predictions, actuals = [], []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_pred = model(X_test)
            test_loss += criterion(y_pred, y_test).item()
            predictions.append(y_pred.numpy())
            actuals.append(y_test.numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # Inverse scale
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    
    plt.plot(actuals[:300], label='Actual')
    plt.plot(predictions[:300], label='Predicted')
    plt.legend()
    plt.show()
